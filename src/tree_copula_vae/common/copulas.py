from enum import Enum
from typing import Type

from tree_copula_vae.torch_copulas.base import PairCopula
from tree_copula_vae.torch_copulas.pair_copulas import (
    BiVariateGaussianCopula,
    BiVariateStudentTCopula,
)


class PairCopulaType(str, Enum):
    """Pair-copula implementations supported by the entropy-based tree VAE."""

    BiVariateGaussianCopula = "BiVariateGaussianCopula"
    BiVariateStudentTCopula = "BiVariateStudentTCopula"

    @property
    def pair_copula_class(self) -> Type[PairCopula]:
        if self is PairCopulaType.BiVariateGaussianCopula:
            return BiVariateGaussianCopula
        if self is PairCopulaType.BiVariateStudentTCopula:
            return BiVariateStudentTCopula
        raise ValueError("Unsupported pair copula type: {}".format(self))