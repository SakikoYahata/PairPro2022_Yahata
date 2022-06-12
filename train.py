import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from statistics import mean

def train(model, dataloader, optimizer, scheduler, device, n_gpu, epoch):
    ce_loss = nn.CrossEntropyLoss()
    train_bar = tqdm(dataloader)
    total_loss = 0
    size = 0
    num_correct = 0
    score = 0
    prediction = []

    for batch_idx, batch in enumerate(train_bar):
        label, batch = batch 
        batch_size = len(batch['input_ids'])
        label = label.to(device)
        batch = {key: value.to(device) for key, value in batch.items()} 
        output = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
        loss = ce_loss(output, label)
        size += batch_size
        predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
        prediction += predictions.tolist()
        batch_num_correct = torch.sum(predictions == label).item()
        num_correct += batch_num_correct
        batch_loss = loss.item() * batch_size
        total_loss += batch_loss
        score = round(batch_num_correct/batch_size, 3)

        if n_gpu > 1:
            loss = loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_bar.set_postfix(
            {
                'lr': scheduler.get_last_lr()[0],
                'loss': round(total_loss / (batch_idx + 1), 3)
            }
        )
        wandb.log(
            {
                'epoch': epoch + size/83136,
                'accuracy_train': score,
                'loss': batch_loss,
                'lr': scheduler.get_last_lr()[0]
            }
        )


    return model