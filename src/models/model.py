from transformers import AutoModelForTokenClassification, AutoModel
import torch.nn as nn

MODEL_CKPT =  "distilbert-base-uncased"
NUM_LABELS = 2

#model = AutoModelForTokenClassification(MODEL_CKPT, num_labels=2)


class SteamModel(nn.Module):
    def __init__(self, name):
        super(SteamModel, self).__init__()
        
        self.base_model = AutoModel.from_pretrained(name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, NUM_LABELS) # output features from bert is 768 
        
    def forward(self, input_ids=None, attention_mask=None,labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
      
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)
        
        return outputs

#model = SteamModel()
#model.to('cuda')