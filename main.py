from argparse import ArgumentParser
from pathlib import Path
import torch
from utils import set_seed, set_device
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import KUCIDataset
from torch.utils.data import DataLoader
from model import BERTPairPro
from train import train
from eval import evaluate


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model", type=str,
        default='/larch/share/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE',
        required=True,
    )
    parser.add_argument(
        "--data_path", type=str,
        default='/mnt/hinoki/karai/KUCI',
        required=True,
    )
    parser.add_argument("--max_seq_len", type=int, default=128, required=True,)
    parser.add_argument("--batch_size", type=int, default=16, required=True,)
    parser.add_argument("--lr", type=float, default=2e-5, required=True,)
    parser.add_argument("--weight_decay", type=float, default=0.01, required=True,)
    parser.add_argument("--seed", type=int, default=0,)
    parser.add_argument("--epoch", type=int, default=3, required=True,)
    parser.add_argument("--warmup_ratio", type=float, default=0.33, required=True)
    parser.add_argument("--save_path", type=str, default="./result/bert", required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    set_seed(args.seed)
    device = set_device(args.gpu_id)
    n_gpu = torch.cuda.device_count()
    print(device, n_gpu)

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model, do_lower_case=False, do_basic_tokenize=False
    )

    train_dataset = KUCIDataset(args.path+'/train.jsonl', tokenizer, args.max_seq_len)
    dev_dataset = KUCIDataset(args.path+'/development.jsonl', tokenizer, args.max_seq_len)

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
        model.train()
        model = train(model, train_dataloader, optimizer, scheduler, device, n_gpu)
        model.eval()
        score, _ = evaluate(model, dev_dataloader, device, n_gpu)



        if best_score is None or score > best_score:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), args.save_path+"/Checkpoint_best.pth")
            best_score = score


if __name__ == "__main__":
    main()