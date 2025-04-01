# core/memory.py 

from collections import deque 
import json
import os 

class ShortTermMemory:
    def __init__(self, maxlen=20):
        self.memory = deque(maxlen=maxlen)

    def store(self, event):
        self.memory.append(event)

    def retrieve_all(self):
        return list(self.memory)
    
    def save(self, path="data/short_term_memory.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.retrieve_all(), f, indent=2)

    def load(self, path="data/short_term_memory.json"):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.memory = deque(json.load(f), maxlen=self.memory.maxlen)