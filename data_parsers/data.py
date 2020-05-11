"""Data module"""
from pathlib import Path
import json
import numpy as np

SWE_TO_ENG = {
    "Fall": "cases",
    "Fall idag": "daily_cases",
    "Döda": "deceased",
    "Döda idag": "daily_deceased",
    "På IVA": "in_icu",
    "På sjukhus": "in_hospital"
}

ENG_TO_SWE = {val: key for (key, val) in SWE_TO_ENG.items()}


class C19Data:
    def __init__(self, cases, daily_cases, deceased, daily_deceased, in_icu,
                 in_hospital):
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
    def from_yaml_dict(yaml):
        """Into object from yaml parsed from website"""
        if not "series" in yaml:
            raise KeyError("Expected key 'series' to be in dict root")
        series = yaml["series"]
        cases = C19Data._parse_series(series, "Fall")
        daily_cases = C19Data._parse_series(series, "Fall idag")
        deceased = C19Data._parse_series(series, "Döda")
        daily_deceased = C19Data._parse_series(series, "Döda idag")
        in_icu = C19Data._parse_series(series, "På IVA")
        in_hospital = C19Data._parse_series(series, "På sjukhus")
        return C19Data(cases, daily_cases, deceased, daily_deceased, in_icu,
                       in_hospital)

    def to_dict(self):
        return {
            data.name: {
                "x": data.x.tolist(),
                "y": data.y.tolist()
            }
            for data in self._to_list()
        }

    def save_to_json(self, filename):
        dict_ = self.to_dict()
        with open(str(filename), 'w') as fp:
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
                return Data(SWE_TO_ENG[name], np.array(x_data),
                            np.array(y_data))

        raise KeyError("Expected key '{}' to be in dict root".format(name))


class Data:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def to_dict(self):
        return {self.name: {"x": self.x, "y": self.y}}
