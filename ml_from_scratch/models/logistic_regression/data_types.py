import io
import logging
import os.path
import tarfile
from dataclasses import dataclass
from typing import Union, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class LRModel:
    def __init__(
            self,
            weights: np.ndarray,
            bias: float,
            classes: Union[np.ndarray, list],
            costs: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.costs = costs
        self.classes = classes

    @property
    def get_costs(self):
        return self.costs

    @property
    def get_bias(self):
        return self.bias

    @property
    def get_weights(self):
        return self.weights

    @property
    def get_classes(self):
        return self.classes

    @staticmethod
    def _load_param(model_path: str, param: str) -> Union[np.ndarray, list, float]:
        param_file_name = f"{param}.npy"

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_path} not found !")

        try:
            with tarfile.open(model_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name == param_file_name and member.isfile():
                        file_obj = tar.extractfile(member)
                        file_content = file_obj.read()
                        res_array = np.load(io.BytesIO(file_content))
                        return res_array
                raise AttributeError(f"'{param_file_name}' not found in {model_path}")
        except (tarfile.TarError, IOError, ValueError) as e:
            logger.error(f"Error loading '{param}' from {model_path}: {e}")

    @classmethod
    def load_model(cls, model_path: str):
        weights = cls._load_param(model_path=model_path, param="weights")
        bias = cls._load_param(model_path=model_path, param="bias")
        classes = cls._load_param(model_path=model_path, param="classes")
        return cls(
            weights=weights,
            bias=bias,
            classes=classes
        )


@dataclass
class TidyData:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    classes: np.ndarray
