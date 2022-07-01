import bz2
from collections import Counter
from typing import Optional
from chess import pgn
import numpy
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from more_itertools import chunked
import itertools as it

# see tokenize docstring for details
VOCAB = [a+b for a,b in it.product('abcdefgh','12345678')]
VOCAB += ['H', 'L', 'W', 'B', 'D', '.']  # . is for padding
VOCAB += ['q','r','b', 'n']  # for pawn promoting
assert len(VOCAB) == len(set(VOCAB)), f'Duplicate vocab entries! {Counter(VOCAB)}'
VOCAB = dict(zip(VOCAB, range(len(VOCAB))))
VOCAB_INV = dict(((y,x) for x,y in VOCAB.items()))


def tokenize(x, vocab=None, pad_to=None):
    """
    First token is H or L for "high rating" and "low rating"
    Last token is W, B, D for who won "white", "black" or "draw" 
    
    >>> tokenize('H e2e4 c7c5 g1f3 W', VOCAB)
    [64, 33, 35, 22, 20, 48, 42, 66]
    >>> x = tokenize('H e2e4 c7c5 g1f3 W', VOCAB, pad_to=12)
    >>> len(x)
    12
    >>> x
    [64, 33, 35, 22, 20, 48, 42, 66, 69, 69, 69, 69]
    """
    if vocab is None:
        vocab = VOCAB
        
    tokens = x.split(' ')
    rv = []

    for token in tokens:
        if len(token) >= 4:
            rv.append(vocab[token[0:2]])
            rv.append(vocab[token[2:4]])
            if len(token) == 5:
                rv.append(vocab[token[4]])
        else:
            assert len(token) == 1, token
            rv.append(vocab[token])
    
    if pad_to:
        if len(rv) > pad_to:
            raise ValueError(f'too many tokens: {x}\n{rv}')
        elif len(rv) < pad_to:
            padding = [vocab['.']] * (pad_to - len(rv))
            rv = rv + padding
    
    return rv

class ChessDataSet(torch.utils.data.Dataset):
    def __init__(self, data: numpy.ndarray, vocab: dict, block_size:int) -> None:
        """

        Args:
            data (numpy.ndarray): 
            vocab (dict): vocabulary
            pad_to (int): amount to pad data samples by
        """
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # block_size + 1 because we inputs/outputs are shifted by 1
        ints = tokenize(self.data[idx], vocab=self.vocab, pad_to=self.block_size+1)
        x = torch.tensor(ints[:-1], dtype=torch.long)
        y = torch.tensor(ints[1:], dtype=torch.long)
        y[y==VOCAB['.']] = -100 # ignore padding in loss
        
        return x,y 
                
class ChessDataModule(LightningDataModule):
    def __init__(self, data_path: numpy.ndarray, vocab: dict, block_size:int, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.vocab = vocab
        self.block_size = block_size
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        data = numpy.load(self.data_path)
        data = data[torch.randperm(len(data), generator=torch.Generator().manual_seed(42))]
        n = len(data)
        n_val = n_test = int(.1 * n)
        n_train = n - n_val - n_test
        
        self.train_data = data[:n_train]
        self.val_data = data[n_train:n_val]
        self.test_data = data[n_train+n_val:]
        
        self.train_dataset = ChessDataSet(self.train_data, vocab=self.vocab, block_size=self.block_size)
        self.val_dataset = ChessDataSet(self.val_data, vocab=self.vocab, block_size=self.block_size)
        self.test_dataset = ChessDataSet(self.test_data, vocab=self.vocab, block_size=self.block_size)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass           
