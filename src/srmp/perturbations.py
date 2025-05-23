import random
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from drs import drs  # type: ignore

from ..dataclass import Dataclass
from ..random import RNGParam, seed


@dataclass
class PerturbWeight(Dataclass):
    amp: float

    def __call__(self, weights: npt.NDArray[np.float64], rng: RNGParam = None):
        random.seed(seed(rng))

        return np.array(
            drs(
                len(weights),
                1,
                list(np.minimum(weights + self.amp, 1).astype(np.float64)),
                list(np.maximum(weights - self.amp, 0).astype(np.float64)),
            )
        )
