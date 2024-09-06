import logging
import os
import pickle
import tarfile
from pathlib import Path
from typing import Dict, Union

import numpy as np
from ml_from_scratch.models.neural_network.forward_propagation import \
    ForwardPropagation
from PIL import Image

logger = logging.getLogger(__name__)


class NeuralNetModel:
    def __init__(
        self,
        parameters: Dict[str, np.ndarray],
        classes: Union[np.ndarray, list],
        activations: Dict[str, str],
    ) -> None:
        self.parameters = parameters
        self.classes = classes
        self.activations = activations

    @property
    def get_activations(self):
        return self.activations

    @property
    def get_parameters(self):
        return self.parameters

    @property
    def get_classes(self):
        return self.classes

    @staticmethod
    def _load_param(model_path: str, param: str) -> Union[np.ndarray, Dict]:
        param_file_name = f"{param}.pkl"

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_path} not found !")

        try:
            with tarfile.open(model_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name == param_file_name and member.isfile():
                        with tar.extractfile(member) as file_obj:
                            res_array = pickle.load(file_obj)  # Corrected this line
                            return res_array
                raise AttributeError(f"'{param_file_name}' not found in {model_path}")
        except (tarfile.TarError, IOError, ValueError) as e:
            logger.error(f"Error loading '{param}' from {model_path}: {e}")
            raise

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """This function is used to predict the results of a  L-layer neural
        network.

        Parameters:
            - X: Data set of examples you would like to label
            - threshold: Probability threshold

        Returns:
            - p: Predictions for the given dataset X
        """

        parameters = self.parameters
        hidden_activation = self.activations["hidden_activation"]
        if not hidden_activation:
            raise KeyError(
                "'hidden_activation' not found in model artefact or not loaded correctly."
            )
        output_activation = self.activations["output_activation"]
        if not output_activation:
            raise KeyError(
                "'hidden_activation' not found in model artefact or not loaded correctly."
            )

        m = X.shape[1]
        p = np.zeros((1, m))

        forward_module = ForwardPropagation(
            hidden_activation=hidden_activation, output_activation=output_activation
        )
        probs, caches = forward_module.model_forward(X=X, model_parameters=parameters)

        for i in range(0, probs.shape[1]):
            if probs[0, i] > threshold:
                p[0, i] = 1
            else:
                p[0, i] = 0
        return p

    def predict_new_image(
        self, image_path: str, number_pixels: int = 64, s: float = 0.5
    ):
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

        pred = self.predict(X=normalized_image, threshold=s)
        pred_class_name = self.classes[int(np.squeeze(pred))].decode("utf-8")
        logger.info(
            f" => y  = {str(np.squeeze(pred))}, model predicted {pred_class_name} for '{image_name}' input."
        )

    @classmethod
    def load_model(cls, model_path: str):
        parameters = cls._load_param(model_path=model_path, param="parameters")
        classes = cls._load_param(model_path=model_path, param="classes")
        activations = cls._load_param(model_path=model_path, param="activations")
        return cls(parameters=parameters, classes=classes, activations=activations)
