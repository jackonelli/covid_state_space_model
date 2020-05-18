"""Types and methods, common to multiple data sources"""
from typing import Iterable


class Serie:
    """Series"""
    def __init__(self, name: str, x: Iterable, y: Iterable):
        self.name = name
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def to_dict(self):
        """Into dict"""
        return {self.name: {"x": self.x, "y": self.y}}
