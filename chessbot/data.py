import bz2
from typing import Optional
from chess import pgn
import numpy
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from more_itertools import chunked
import itertools as it

VOCAB = [a+b for a,b in it.product('abcdefgh','12345678')]
VOCAB += [' ', 'A','B', '2', '1', '0']
VOCAB = dict(zip(VOCAB, range(len(VOCAB))))

def tokenize(x, vocab):
    """
    First token is A or B which we map to 
    Last token is w/l/d
    
    >>> list(tokenize('B e2e4 c7c5 g1f3 1', VOCAB))
    [66, 64, 33, 35, 64, 22, 20, 64, 48, 42, 64, 68]
    """
    tokens = x.split(' ')
    yield vocab[tokens[0]]
    yield vocab[' ']

    for token in tokens[1:-1]:
        yield vocab[token[0:2]]
        yield vocab[token[2:4]]
        yield vocab[' ']
    
    yield vocab[tokens[-1]]
    

class ChessDataSet(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
                
class ChessDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path

    def setup(self, stage: Optional[str] = None):
        data = numpy.load(self.data_path)
        n = len(data)
        n_val = n_test = int(.1 * n)
        n_train = n - n_val - n_test
        
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(data, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
        
        self.train_dataset = ChessDataSet(self.train_data)
        self.val_dataset = ChessDataSet(self.val_data)
        self.test_dataset = ChessDataSet(self.test_data)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass           
