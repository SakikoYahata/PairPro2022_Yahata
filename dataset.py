from torch.utils.data import Dataset
import json

class KUCIDataset(Dataset):
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