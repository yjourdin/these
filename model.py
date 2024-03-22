from typing import Literal

from abstract_model import Model
from rmp.model import RMPModel
from srmp.model import SRMPModel

ModelType = Literal["SRMP", "RMP"]


def import_model(s: str) -> Model:
    if "capacities" in s:
        return RMPModel.from_json(s)
    elif "weights" in s:
        return SRMPModel.from_json(s)
    else:
        raise ValueError("model is not a valid model")
