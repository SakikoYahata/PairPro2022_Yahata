# %%
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
import csv

# %%
class CFG():
    pretrained_model = '/larch/share/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE'
    path = '/mnt/hinoki/karai/KUCI'
    max_seq_len = 128
    batch_size = 16
    lr = 2e-5
    weight_decay = 0.01
    seed = 0
    epoch = 3
    warmup_ratio = 0.033
    save_path = "./result/bert"

# %%
save_path = Path(CFG.save_path)
save_path.mkdir(exist_ok=True, parents=True)

# %%
import random
import numpy as np
import torch
import os
import re


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpuid: str) -> torch.device:
    if gpuid and torch.cuda.is_available():
        assert re.fullmatch(r"[0-7]", gpuid) is not None, "invalid way to specify gpuid"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
        device = torch.device(f"cuda:{gpuid}")
    else:
        device = torch.device("cpu")

    return device

# %%
set_seed(CFG.seed)
device = set_device("0")
print(device)

# %%
tokenizer = BertTokenizer.from_pretrained(
    CFG.pretrained_model, do_lower_case=False, do_basic_tokenize=False
)

print(tokenizer.tokenize('今日は曇りです'))
print(tokenizer([['今日は曇りです', '明日は晴れるといいな'],['明日はピクニックです', 'お弁当楽しみだな']]))

# %%
class PairPro(Dataset):
    def __init__(self, path, tokenizer, max_seq_len, is_test=False):
        self.is_test = is_test
        self.label2int = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        self.labels, self.contexts, self.choices = self.load(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
    
    def load(self, path):
        label_list = []
        context_list = []
        choice_list = []
        with open(path) as f:
            for i, line in enumerate(f):
                problem = json.loads(line)
                if self.is_test:
                    label_list.append(-1)
                else:
                    label_list.append(self.label2int[problem['label']])
                context_list.append(problem['context'])
                choice_list.append([problem['choice_a'],problem['choice_b'],problem['choice_c'],problem['choice_d']])
        assert len(label_list) == len(context_list), "長さが違います"
        return label_list, context_list, choice_list

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input_list = []
        for choice in self.choices[idx]:
            input_list.append([self.contexts[idx], choice])    
        return self.labels[idx], self.tokenizer(
            input_list,
            max_length=self.max_seq_len, 
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            )

# %%
traindataset = PairPro(CFG.path+'/train.jsonl', tokenizer, CFG.max_seq_len)
print(traindataset[0])  # pairの要素を参照すると__getitem__が呼び出される
devdataset = PairPro(CFG.path+'/development.jsonl', tokenizer, CFG.max_seq_len)
testdataset = PairPro(CFG.path+'/test.jsonl', tokenizer, CFG.max_seq_len, is_test=True)

# %%
traindataloader = DataLoader(traindataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
devataloader = DataLoader(devdataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
testdataloader = DataLoader(testdataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

# %%
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

# %%

model = BERTPairPro(CFG.pretrained_model)
model = model.to(device)

# %%
# for i, batch in enumerate(PairProdataloader):
#     #print(batch)
#     label, batch = batch
#     if i == 0:
#         output = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
#         print(output)
#         break

# %%
optimizer = AdamW(
    model.parameters(),
    lr=CFG.lr,
    weight_decay=CFG.weight_decay,
    no_deprecation_warning=True
)

# %%
num_training_steps = len(traindataloader) * CFG.epoch
num_warmup_steps = num_training_steps * CFG.warmup_ratio
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_training_steps=num_training_steps,
    num_warmup_steps=num_warmup_steps
)

ce_loss = nn.CrossEntropyLoss()

# %%
best_score = None
for epoch in range(CFG.epoch):
    total_loss = 0
    model.train()
    train_bar = tqdm(traindataloader)
    for batch_idx, batch in enumerate(train_bar):
        label, batch = batch 
        batch_size = len(batch['input_ids'])
        label = label.to(device)
        batch = {key: value.to(device) for key, value in batch.items()} 
        output = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
        loss = ce_loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch_size
        train_bar.set_postfix(
            {
                'lr': scheduler.get_last_lr()[0],
                'loss': round(total_loss / (batch_idx + 1), 3)
            }
        )
    

    model.eval()
    with torch.no_grad():
        dev_bar = tqdm(devataloader)
        num_correct = 0
        total_loss = 0
        size = 0 
        for batch_idx, batch in enumerate(dev_bar):
            label, batch = batch 
            batch_size = len(batch['input_ids'])
            label = label.to(device)
            batch = {key: value.to(device) for key, value in batch.items()} 
            output = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            loss = ce_loss(output, label)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            num_correct += torch.sum(predictions == label).item()
            size += batch_size
            total_loss += loss.item() * batch_size
            score = round(num_correct/size, 3)

            dev_bar.set_postfix(
            {
                'size': size,
                'accuracy': score,
                'loss': round(total_loss / (batch_idx + 1), 3)
            }
        )
    
    score = round(num_correct/size, 3)
    if best_score is None or score > best_score:
        torch.save(model.state_dict(), CFG.save_path+"/Checkpoint_best.pth")
        best_score = score



# %%
model = BERTPairPro(CFG.pretrained_model)
model = model.to(device)

# %%
state_dict = torch.load(CFG.save_path+"/Checkpoint_best.pth", map_location=device)
model.load_state_dict(state_dict)   

# %%
prediction = []
model.eval()
with torch.no_grad():
    test_bar = tqdm(testdataloader)
    num_correct = 0
    total_loss = 0
    size = 0 
    for batch_idx, batch in enumerate(test_bar):
        label, batch = batch 
        batch_size = len(batch['input_ids'])
        # label = label.to(device)
        batch = {key: value.to(device) for key, value in batch.items()} 
        output = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
        # loss = ce_loss(output, label)
        predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
        size += batch_size
        #total_loss += loss.item() * batch_size
        prediction += predictions.tolist()
    
    int2label = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    prediction = [int2label[pre] for pre in prediction]
    print(prediction)

    #     #dev_bar.set_postfix(
    #     {
    #         'size': size,
    #         #'accuracy': round(num_correct/size, 3),
    #         #'loss': round(total_loss / (batch_idx + 1), 3)
    #     }
    # )

    

# %%
with open(CFG.save_path+"/test_prediction.csv", "w") as f:
  csv.writer(f).writerows(prediction)
f.close()


