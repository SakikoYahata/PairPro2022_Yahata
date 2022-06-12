from argparse import ArgumentParser
from pathlib import Path
import torch
from utils import set_seed, set_device
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from dataset import KUCIDataset
from torch.utils.data import DataLoader
from model import BERTPairPro
from train import train
from eval import evaluate
import os
import wandb


wandb.init(project="pair-programming")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model", type=str,
        default='/larch/share/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE',
    )
    parser.add_argument(
        "--data_path", type=str,
        default='/mnt/hinoki/karai/KUCI',
    )
    parser.add_argument("--max_seq_len", type=int, default=128,)
    parser.add_argument("--batch_size", type=int, default=16,)
    parser.add_argument("--lr", type=float, default=2e-5,)
    parser.add_argument("--weight_decay", type=float, default=0.01,)
    parser.add_argument("--seed", type=int, default=0,)
    parser.add_argument("--epoch", type=int, default=3,)
    parser.add_argument("--warmup_ratio", type=float, default=0.33, )
    parser.add_argument("--save_path", type=str, default="./result/bert", )
    parser.add_argument("--gpu_id", type=str, )
    args = parser.parse_args()

    wandb.log(
        {
            '--max_seq_len': args.max_seq_len, 
            '--batch_size': args.batch_size,
            '--lr': args.lr,
            '--weight_decay': args.weight_decay,
            '--seed': args.seed,
            '--epoch': args.epoch,
            '--warmup_ratio': args.epoch
        }
    )
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    set_seed(args.seed)
    # device = set_device(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    n_gpu = torch.cuda.device_count()
    print(device, n_gpu)

    if args.pretrained_model == '/larch/share/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE':
        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model, do_lower_case=False, do_basic_tokenize=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model, do_lower_case=False, do_basic_tokenize=False
        )    

    train_dataset = KUCIDataset(args.data_path+'/train.jsonl', tokenizer, args.max_seq_len)
    dev_dataset = KUCIDataset(args.data_path+'/development.jsonl', tokenizer, args.max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = BERTPairPro(args.pretrained_model)
    model = model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        no_deprecation_warning=True
    )

    num_training_steps = len(train_dataloader) * args.epoch
    num_warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    best_score = None

    for epoch in range(args.epoch):
        if epoch == 0:
            model.eval()
            score, _ = evaluate(model, dev_dataloader, device, n_gpu, -1)
        model.train()
        model = train(model, train_dataloader, optimizer, scheduler, device, n_gpu, epoch)
        model.eval()
        score, _ = evaluate(model, dev_dataloader, device, n_gpu, epoch)

        if best_score is None or score > best_score:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), args.save_path+"/Checkpoint_best.pth")
            best_score = score


    wandb.save("pair-programming.h5")


if __name__ == "__main__":
    main()