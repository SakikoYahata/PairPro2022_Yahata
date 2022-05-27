from torch import nn
from tqdm import tqdm


def train(model, dataloader, optimizer, scheduler, device, n_gpu):
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0
    train_bar = tqdm(dataloader)
    for batch_idx, batch in enumerate(train_bar):
        label, batch = batch 
        batch_size = len(batch['input_ids'])
        label = label.to(device)
        batch = {key: value.to(device) for key, value in batch.items()} 
        output = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
        loss = ce_loss(output, label)
    
        if n_gpu > 1:
            loss = loss.mean()

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
    return model