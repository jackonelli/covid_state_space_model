"""Data module"""
from typing import Union, Dict
from pathlib import Path
import json
from datetime import datetime
from collections import OrderedDict
import numpy as np
from data.common import Serie

SWE_TO_ENG = OrderedDict({
    "Fall": "cases",
    "Fall idag": "daily_cases",
    "Döda": "deceased",
    "Döda idag": "daily_deceased",
    "På IVA": "in_icu",
    "På sjukhus": "in_hospital"
})

ENG_TO_SWE = {val: key for (key, val) in SWE_TO_ENG.items()}


class C19Data:
    """C19Data"""
    def __init__(self, cases, daily_cases, deceased, daily_deceased, in_icu,
                 in_hospital):
        self.created = datetime.now()
        self.cases = cases
        self.daily_cases = daily_cases
        self.deceased = deceased
        self.daily_deceased = daily_deceased
        self.in_icu = in_icu
        self.in_hospital = in_hospital

    def _to_list(self):
        return [
            self.cases, self.daily_cases, self.deceased, self.daily_deceased,
            self.in_icu, self.in_hospital
        ]

    @staticmethod
    def from_yaml_dict(yaml: Dict):
        """Into object from yaml parsed from website"""
        if "series" not in yaml:
            raise KeyError("Expected key 'series' to be in dict root")
        series = yaml["series"]
        return C19Data(*[
            C19Data._parse_series(series, serie_name)
            for serie_name in SWE_TO_ENG
        ])

    def to_dict(self):
        """To dictionary"""
        dict_ = {
            data.name: {
                "x": data.x.tolist(),
                "y": data.y.tolist()
            }
            for data in self._to_list()
        }
        dict_["created"] = self.created.strftime("%Y-%m-%dT%H:%M:%S")
        return dict_

    def save_to_json(self, filename: Union[str, Path]):
        """Write data as dict to json file"""
        dict_ = self.to_dict()
        with open(str(filename), "w") as fp:
            json.dump(dict_, fp, indent=4, sort_keys=True)

    @staticmethod
    def _parse_series(series, name):
        for serie in series:
            if serie["name"] == name:
                x_data = []
                y_data = []
                for point in serie["data"]:
                    x_data.append(point["x"])
                    y_data.append(point["y"])
                return Serie(SWE_TO_ENG[name], np.array(x_data),
                             np.array(y_data))

        raise KeyError("Expected key '{}' to be in dict root".format(name))