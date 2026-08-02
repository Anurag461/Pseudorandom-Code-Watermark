import json
import aiohttp
import os
import datasets
from typing import List, Dict
from abc import ABC, abstractmethod

class HFDataset:
    def __init__(self, *args, **kwargs):
        self.dataset = datasets.load_dataset(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.dataset[idx]

class Task(ABC):
    def __init__(self, start=0, stop=None, step=1):
        assert start >=  0
        assert stop is None or stop > start
        assert step >= 1

        self.start = start
        self.stop = stop
        self.step = step

    @abstractmethod
    def num_examples(self):
        pass

    @abstractmethod
    def get_example(self, index):
        pass

    @abstractmethod
    def evaluate(self, problem, completion):
        pass
    
    def __len__(self):
        stop = self.stop if self.stop is not None else self.num_examples()
        span = stop - self.start
        length = (span + self.step -1) // self.step
        assert length >= 0, "len cannot be -ve"
        return length

        
