"""Birch

Types and methods to transform data to a suitable format for birch
"""
from typing import Dict, List
from data.common import Serie


def into_birch_dict(series: List["Serie"], start_state: Dict,
                    prior_params: Dict) -> Dict:
    """Transform serie into birch dict

    TODO: Properly handle states not being present at different timesteps
    """

    birch_dict = {}
    birch_dict["theta"] = prior_params
    birch_dict["x"] = [start_state]
    for time_k in range(len(series[0])):
        birch_dict["x"].append({
            state_component.name: state_component.y[time_k]
            for state_component in series
        })
    return birch_dict
