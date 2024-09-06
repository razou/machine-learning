import io
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

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
            raise

    @staticmethod
    def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the sigmoid function for z :param z: scalar or vector or
        matrix :return: 1 /  (1 + exp(-z))"""
        s = 1 / (1 + np.exp(-z))
        return s

    def predict(self, data: np.ndarray, s: float = 0.5) -> np.ndarray:
        """
        Parameters:
            - data: Test data matrix (to use for prediction). Matrix of size (number_pixels * number_pixels * 3, ...)
            - s (float): Threshold (to decide whether in which class to classify a given sample regarding the proba score)
        Returns:
            An Array of predictions
        """

        num_samples = data.shape[1]
        y_pred = np.zeros((1, num_samples))
        weights = self.weights.reshape(data.shape[0], 1)

        y_hat = self.sigmoid(np.dot(weights.T, data) + self.bias)

        for i in range(y_hat.shape[1]):
            if y_hat[0, i] > s:
                y_pred[0, i] = 1
            else:
                y_pred[0, i] = 0

        return y_pred

    def predict_new_image(self, image_path: str, s: float, number_pixels: int = 64):
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"{image_path} not found !")

        image_name = os.path.basename(image_path)

        logger.debug("Image to vector transformation")
        """PNG images often have an alpha channel (transparency) in addition to
        the RGB color channels, which can affect how the image data is
        interpreted and processed.

        To handle PNG images correctly alongside JPEG images, we specify
        the mode as 'RGB' when loading images to ensure that the image
        is loaded without the alpha channel.
        """
        image = (
            Image.open(image_path).convert("RGB").resize((number_pixels, number_pixels))
        )
        image2array = np.array(image)
        logger.debug("Image normalization")
        normalized_image = image2array / 255.0
        normalized_image = normalized_image.reshape(
            (1, number_pixels * number_pixels * 3)
        ).T
        pred = self.predict(data=normalized_image, s=s)

        pred_class_name = self.classes[int(np.squeeze(pred))].decode("utf-8")
        logger.info(
            f" => y  = {str(np.squeeze(pred))}, model predicted '{pred_class_name}' for '{image_name}' input."
        )

    @classmethod
    def load_model(cls, model_path: str):
        weights = cls._load_param(model_path=model_path, param="weights")
        bias = cls._load_param(model_path=model_path, param="bias")
        classes = cls._load_param(model_path=model_path, param="classes")
        return cls(weights=weights, bias=bias, classes=classes)
