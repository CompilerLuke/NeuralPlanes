from torch.utils.data import Dataset, DataLoader
import heapq
import itertools
from NeuralPlanes.camera import Camera
from NeuralPlanes.utils import select_device
from dataclasses import dataclass
import numpy as np
import torch
import cv2

@dataclass
class DataloaderConf:
    batch_size: int = 16
    cache_size: int = 32
    num_workers: int = 4
    device: torch.device = select_device()

def nested_shape(value):
    if isinstance(value, np.ndarray):
        return value.shape
    if isinstance(value, torch.Tensor):
        return value.shape 
    if isinstance(value, list):
        return [nested_shape(x) for x in value]
    if isinstance(value, dict):
        return {k: nested_shape(v) for k,v in value.items()}
    return value 

class resize:
    def __init__(self, max_size):
        self.max_size = max_size 

    def __call__(self, data):
        max_size = self.max_size
        if not "image" in data:
            return data
        
        image = data["image"]

        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            scale = max_size / max(height,width)
            image = cv2.resize(image, (int(width*scale),int(height*scale)))
        else:
            height, width = image.shape[1:3]
            scale = max_size / max(height,width)
            image = torch.nn.functional.interpolate(image[None], (int(height),int(width)), mode='bilinear').squeeze()

        if "camera" in data:
            camera = data["camera"]
            return { **data, "image": image, "camera": camera.scale(scale)}
        return { **data, "image": image }

class compose:
    def __init__(self, fns):
        self.fns = fns 

    def __call__(self, data):
        for fn in self.fns:
            data = fn(data)
        return data

class MapDataset:
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, item):
        return self.dataset1[item], self.dataset2[item]

class ReorderDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset 
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

class SequentialLoader:
    def __init__(self, dataset, accesses, conf: DataloaderConf, collate_fn=None, apply_fn=None):
        cache_size = conf.cache_size
        batch_size = conf.batch_size

        last_access = {access: 100000 for access in accesses}
        next_access = np.zeros(len(accesses), dtype=int)
        for i in reversed(range(len(accesses))):
            access = accesses[i]
            next_access[i] = last_access[access]
            last_access[access] = i 

        cache = {}
        latest_future_access_heap = []
        
        load_cache_slot_index = []
        load_items = []
        batch_number = []
        cache_slot_index = []

        free_slots = []
        filled = False

        for i, access in enumerate(accesses):
            if access in cache:
                hit = True
                cache_slot = cache[access]
            else:
                hit = False
                if not filled:
                    cache_slot = len(cache)
                    filled = len(cache)+1 >= cache_size+batch_size
                elif len(load_items) % batch_size == 0:
                    while len(cache) >= cache_size:
                        priority, cache_slot, item = heapq.heappop(latest_future_access_heap)
                        del cache[item]
                        free_slots.append(cache_slot)
                    cache_slot = free_slots.pop()
                else:
                    cache_slot = free_slots.pop()
                cache[access] = cache_slot
                load_cache_slot_index.append(cache_slot)
                load_items.append(access)
                heapq.heappush(latest_future_access_heap, (-next_access[i], cache_slot, access))

            batch_number.append((len(load_items)-1) // conf.batch_size)
            cache_slot_index.append(cache_slot)

        self.apply_fn = apply_fn
        self.current_access = 0
        self.current_batch_number = -1
        self.batch_number = batch_number
        self.dataloader = DataLoader(ReorderDataset(dataset=dataset, indices=load_items), prefetch_factor=cache_size//batch_size, num_workers=conf.num_workers, collate_fn=collate_fn, batch_size=conf.batch_size)
        self.iterator = None
        self.load_items = load_items
        self.load_cache_slot_index = load_cache_slot_index
        self.accesses = accesses
        self.cache_size = cache_size + batch_size
        self.batch_size = conf.batch_size 
        self.cache = [None for i in range(self.cache_size)]
        self.cache_slot_index = cache_slot_index

    def fetch_next_batch(self):
        current_batch_number = self.current_batch_number+1
        data = next(self.iterator)
        if self.apply_fn:
            data = self.apply_fn(data)
        
        for j in range(self.batch_size):
            idx = self.batch_size*current_batch_number + j
            if idx >= len(self.load_cache_slot_index):
                break

            data_j = None
            if isinstance(data, dict):
                data_j = {k: v[j] for k, v in data.items()}
            else:
                data_j = data[j]
            self.cache[self.load_cache_slot_index[idx]] = data_j

        self.current_batch_number = self.current_batch_number+1

    def __getitem__(self, access):
        if access != self.accesses[self.current_access]:
            raise "Access pattern mismatch"
        idx = self.current_access
        batch_number = self.batch_number[idx]
        if batch_number == 0 and self.current_batch_number != 0:
            del self.iterator
            self.iterator = self.dataloader.__iter__()
            self.fetch_next_batch()
        if batch_number > self.current_batch_number:
            self.fetch_next_batch()

        self.current_access = (self.current_access + 1) % len(self.accesses)
        return self.cache[self.cache_slot_index[idx]]

