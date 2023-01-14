import torch
import evaluate
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore

from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Value, load_from_disk
from transformers import AutoModelForSequenceClassification,Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

from src.data.make_dataset import ReviewDataset
from src.models.model import SteamModel
from src.models.config import SteamConfig



SAMPLE_SIZE = 1000
MODEL_CKPT =  "distilbert-base-uncased"
NUM_LABELS = 2
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

cs = ConfigStore.instance()
cs.store(name='steam_config', node = SteamConfig)

@hydra.main(config_path='conf', config_name='config.yaml')
def main(cfg:SteamConfig):

    processed_data = ReviewDataset(cfg.paths.in_folder, cfg.paths.out_folder, name=cfg.params.model_ckpt, sample_size=cfg.params.sample_size, force=False)
    emotions_encoded = processed_data.processed
    tokenizer = processed_data.tokenizer

    model = SteamModel(cfg.params.model_ckpt, cfg.params.num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #config = AutoConfig.from_pretrained(MODEL_CKPT)
    #model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)

    logging_steps = len(emotions_encoded["train"]) // cfg.params.batch_size
    model_name = f"{cfg.params.model_ckpt}-finetuned-Steam"

    #model(input, MODEL_CKPT)

    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=cfg.params.epochs,
                                    learning_rate=cfg.params.lr,
                                    per_device_train_batch_size=cfg.params.batch_size,
                                    per_device_eval_batch_size=cfg.params.batch_size,
                                    weight_decay=cfg.params.weight_decay,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error")
                                    
    trainer = Trainer(model=model, args=training_args, 
                    #compute_metrics=compute_metrics,
                    train_dataset=emotions_encoded["train"],
                    eval_dataset=emotions_encoded["valid"],
                    tokenizer=tokenizer)

    # model is stuck on training, still gotta find out why
    trainer.train()
    #trainer.save_model('models/')

if __name__ == "__main__":
    main()