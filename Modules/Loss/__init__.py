from .BendingEnergy import (BendingEnergyMetric, RBFBendingEnergyLoss,
                            RBFBendingEnergyLossA)
from .CrossCorrelation import (LocalCrossCorrelation2D, LocalCrossCorrelation3D,LocalCrossCorrelation2D_Uncertainty,
                               WeightedLocalCrossCorrelation2D)
from .DiceCoefficient import DiceCoefficient, DiceCoefficientAll
from .Distance import MaxMinPointDist, SurfaceDistanceFromSeg
from .Jacobian import JacobianDeterminantLoss, JacobianDeterminantMetric
from .MeanSquareError import MeanSquareError

from .OtherLoss import gradient_loss

LOSSDICT = {
    'LCC': LocalCrossCorrelation2D,
    'ULCC': LocalCrossCorrelation2D_Uncertainty,
    'WLCC': WeightedLocalCrossCorrelation2D,
    'MSE': MeanSquareError
}
