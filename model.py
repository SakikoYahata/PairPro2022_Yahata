import torch.nn as nn
from transformers import BertModel

class BERTPairPro(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model, output_attentions=False
        )
        self.linear = nn.Linear(
            self.bert.config.hidden_size, 1
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size, n_choice, seq_len = input_ids.shape     # 16,4,128
        input_ids = input_ids.view(batch_size*n_choice, seq_len)        # 64,128
        attention_mask = attention_mask.view(batch_size*n_choice, seq_len)
        token_type_ids = token_type_ids.view(batch_size*n_choice, seq_len)
        # print(input_ids.shape)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = output['pooler_output']   # 64, 768
        output = self.linear(cls)   # 64, 1
        output = output.squeeze(1).view(batch_size, n_choice)
        return output