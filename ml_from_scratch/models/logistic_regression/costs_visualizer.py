from typing import Union, List
import numpy as np
from matplotlib import pyplot as plt


def plot_costs(costs: Union[np.ndarray, List[float]], learning_rate: float):
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Num Iterations (per hundreds)')
    plt.title(r"Learning rate ($\alpha$) =" + str(learning_rate))
    plt.show()