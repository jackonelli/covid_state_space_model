"""Data module"""
from typing import Union, Dict
from pathlib import Path
import json
from datetime import datetime
from collections import OrderedDict
import numpy as np
from data.common import Serie


class DailyIcu:
    """DailyIcu cases"""
    def __init__(self, daily_icu: Serie):
        self.created = datetime.now()
        self.daily_icu = daily_icu

    def __len__(self):
        return len(self.daily_icu)

    @staticmethod
    def from_csv(data_path: Path) -> "DailyIcu":
        """Load data from csv file"""
        daily_icu = np.loadtxt(data_path, dtype=int, delimiter=";", skiprows=1)

        return DailyIcu(
            Serie("daily_icu",
                  np.arange(daily_icu.shape[0]),
                  daily_icu))

    def _to_numpy(self):
        return self.daily_icu.y

    def save_to_csv(self, filename: Union[str, Path]):
        """Write data to CSV"""
        np.savetxt(str(filename),
                   self.daily_icu.y,
                   delimiter=";",
                   header=self.daily_icu.name)
