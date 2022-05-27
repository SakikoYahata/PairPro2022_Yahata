import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


def evaluate(model, dataloader, device, n_gpu):
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        dev_bar = tqdm(dataloader)
        num_correct = 0
        total_loss = 0
        size = 0 
        prediction = []
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

            prediction += predictions.tolist()

            if n_gpu > 1:
                loss = loss.mean()

            dev_bar.set_postfix(
                {
                    'size': size,
                    'accuracy': score,
                    'loss': round(total_loss / (batch_idx + 1), 3)
                }
            )
        
        int2label = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        prediction = [int2label[pre] for pre in prediction]
        
        score = round(num_correct/size, 3)
        return score, prediction