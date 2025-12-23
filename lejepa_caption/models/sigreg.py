import torch
import torch.nn as nn

from lejepa.univariate import EppsPulley
from lejepa.multivariate import SlicingUnivariateTest


class SIGRegLoss(nn.Module):
    """
    SIGReg regularizer built from univariate Epps-Pulley and multivariate slicing.

    Flattens all leading dimensions into the sample axis, then applies the
    characteristic-function-based Gaussianity test to encourage isotropy and
    dispersion without explicit negatives.
    """

    def __init__(self, n_points: int = 17, num_slices: int = 128, reduction: str = "mean"):
        super().__init__()
        self.test = SlicingUnivariateTest(
            univariate_test=EppsPulley(n_points=n_points),
            num_slices=num_slices,
            reduction=reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collapse sequence/batch dims to (N, D) for the slicing test.
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
        return self.test(x)
