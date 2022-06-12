from argparse import ArgumentParser
from eval import evaluate
from pathlib import Path
import torch
from utils import set_seed, set_device
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from dataset import KUCIDataset
from torch.utils.data import DataLoader
from model import BERTPairPro
import csv
import os

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model", type=str,
        default='/larch/share/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE',
    )
    parser.add_argument(
        "--data_path", type=str, default='/mnt/hinoki/karai/KUCI',
    )
    parser.add_argument("--max_seq_len", type=int, default=128,)
    parser.add_argument("--batch_size", type=int, default=16,)
    #parser.add_argument("--weight_decay", type=float, default=0.01,)
    parser.add_argument("--seed", type=int, default=0,)
    #parser.add_argument("--epoch", type=int, default=3,)
    #parser.add_argument("--warmup_ratio", type=float, default=0.33,)
    parser.add_argument("--save_path", type=str, default="./result/bert",)
    parser.add_argument("--gpu_id", type=str,)
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    set_seed(args.seed)
    # device = set_device(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    
    n_gpu = torch.cuda.device_count()
    print(device)
    print(n_gpu)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model, do_lower_case=False, do_basic_tokenize=False
    )

    dev_dataset = KUCIDataset(args.data_path+'/development.jsonl', tokenizer, args.max_seq_len, is_test=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = BERTPairPro(args.pretrained_model)
    model = model.to(device)
    state_dict = torch.load(args.save_path+"/Checkpoint_best.pth", map_location=device)
    model.load_state_dict(state_dict)  
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    _, prediction = evaluate(model, dev_dataloader, device, n_gpu, is_test=True)

    with open(args.save_path+"/dev_prediction.csv", "w") as f:
        csv.writer(f).writerows(prediction)
    f.close()


if __name__ == "__main__":
    main()