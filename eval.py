import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import wandb


def evaluate(model, dataloader, device, n_gpu, epoch, is_test=False):
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        dev_bar = tqdm(dataloader)
        total_loss = 0
        size = 0 
        num_correct = 0
        score = 0
        prediction = []
        for batch_idx, batch in enumerate(dev_bar):
            label, batch = batch 
            batch_size = len(batch['input_ids'])
            if not is_test:
                label = label.to(device)
            batch = {key: value.to(device) for key, value in batch.items()} 
            output = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            if not is_test:
                loss = ce_loss(output, label)
            size += batch_size
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            prediction += predictions.tolist()
            if not is_test:
                num_correct += torch.sum(predictions == label).item()
                total_loss += loss.item() * batch_size
                score = round(num_correct/size, 3)

                if n_gpu > 1:
                    loss = loss.mean()

                dev_bar.set_postfix(
                    {
                        'size': size,
                        'accuracy': score,
                        'loss': round(total_loss / (batch_idx + 1), 3)
                    }
                )
        if not is_test:
            wandb.log(
                {
                'epoch': epoch+1,
                'accuracy': score,
                'loss_dev': total_loss / (batch_idx + 1)
                }
            )
            
        int2label = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        prediction = [int2label[pre] for pre in prediction]
        
        if not is_test:
            score = round(num_correct/size, 3)
        else:
            print(prediction)
    
        return score, prediction