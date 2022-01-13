import numpy as np


def Kalman_gain(H: np.ndarray, Pf: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Computes the Kalman Gain Given the observation matrix, the prior covariance matrix and the error covariance matrix error R
    :param H: Linearized observation operator
    :type H: np.ndarray
    :param Pf: Covariance matrix of the prior error
    :type Pf: np.ndarray
    :param R: Covariance matrix of the observation errors
    :type R: np.ndarray
    :return: Kalman Gain
    :rtype: np.ndarray
    """
    return np.linalg.multi_dot(
        [
            Pf,
            H.T,
            np.linalg.inv(np.linalg.multi_dot([H, Pf, H.T]) + R),
        ]
    )
