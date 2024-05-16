import unittest

import numpy as np
from ml_from_scratch.models.logistic_regression impport sigmoid

def test_sigmoid_test():
    x = np.array([0, 2])
    output = sigmoid(x)
    assert type(output) == np.ndarray, "Wrong type. Expected np.ndarray"
    assert np.allclose(output, [0.5, 0.88079708]), f"Wrong value. {output} != [0.5, 0.88079708]"
    output = sigmoid(1)
    assert np.allclose(output, 0.7310585), f"Wrong value. {output} != 0.7310585"
    print('\033[92mAll tests passed!')


if __name__ == '__main__':
    test_sigmoid_test()
