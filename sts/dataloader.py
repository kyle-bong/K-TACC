import re
import pandas as pd
from tqdm.auto import tqdm
import transformers
import torch
import pytorch_lightning as pl
from ast import literal_eval

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]).unsqueeze(0)
        
    def __len__(self):
        return len(self.inputs)
    
class Dataloader(pl.LightningDataModule):
    def __init__(self, cfg, idx=None):
        super().__init__()
        self.model_name = cfg.model.model_name
        self.batch_size = cfg.train.batch_size
        self.shuffle = cfg.data.shuffle
        self.max_length = cfg.data.max_length

        self.train_path = cfg.path.train_path
        self.dev_path = cfg.path.dev_path
        self.test_path = cfg.path.test_path

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_length)
        self.target_columns = ['labels']
        self.delete_columns = ['guid']
        self.text_columns = ['sentence1', 'sentence2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = self.tokenizer.sep_token.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text,
                add_special_tokens=True, 
                max_length=self.max_length, 
                padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data
    
    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        targets = [x[0]['label'] for x in data[self.target_columns].values.tolist()]
        inputs = self.tokenizing(data)
        return inputs, targets
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_json(self.train_path)
            dev_data = pd.read_json(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)
            dev_inputs, dev_targets = self.preprocessing(dev_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.dev_dataset = Dataset(dev_inputs, dev_targets)

        else:
            test_data = pd.read_json(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False)
    
    
    