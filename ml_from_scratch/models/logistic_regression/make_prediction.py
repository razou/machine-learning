import argparse
import logging
import os.path
from pathlib import Path

import numpy as np
from PIL import Image

from constants import ROOT_DIR
from data_types import LRModel
from lr_model_predict import predict

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_registry', type=str, default="Artefacts", help="Model registry.")
    parser.add_argument('--model_dir', type=str, default="lr_model", help="Model dir")
    parser.add_argument('--model_name', type=str, default="lr", help="Model name")
    args = parser.parse_args()
    return args


def predict_new_image(
        image_path: str,
        w: np.ndarray,
        b: float,
        s: float,
        target_classes: list,
        number_pixels: int = 64
):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"{image_path} not found !")

    image_name = os.path.basename(image_path)

    logger.debug("Image to vector transformation")
    """
    PNG images often have an alpha channel (transparency) in addition to the RGB color channels, 
    which can affect how the image data is interpreted and processed.
    To handle PNG images correctly alongside JPEG images, we specify the mode as 'RGB' when loading images
    to ensure that the image is loaded without the alpha channel.
    
    """
    image = Image.open(image_path).convert("RGB").resize((number_pixels, number_pixels))
    image2array = np.array(image)
    logger.debug("Image normalization")
    normalized_image = image2array / 255.
    normalized_image = normalized_image.reshape((1, number_pixels * number_pixels * 3)).T
    pred = predict(weights=w, bias=b, data=normalized_image, s=s)

    pred_class_name = target_classes[int(np.squeeze(pred))].decode('utf-8')
    print(f" => y  = {str(np.squeeze(pred))}, the LR classifier predicted "
          f"'{pred_class_name}' for '{image_name}' picture.")


def main():
    parsed_args = _parse_args()
    registry_dir_name = parsed_args.model_registry
    model_dir_name = parsed_args.model_dir
    model_artefact_name = parsed_args.model_name

    registry_dir_path = os.path.join(ROOT_DIR, registry_dir_name)
    model_dir_path = os.path.join(registry_dir_path, model_dir_name)
    model_artefact_path = os.path.join(model_dir_path, f"{model_artefact_name}.tar.gz")

    lr_model = LRModel.load_model(model_path=model_artefact_path)
    weights = lr_model.weights
    bias = lr_model.bias
    classes = lr_model.classes

    images_dir = os.path.join(ROOT_DIR, "DATASETS/images")
    for img in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img)
        try:
            predict_new_image(image_path=img_path, w=weights, b=bias, target_classes=classes, s=0.5)
        except Exception as e:
            print(f"Unable to predict image '{img}' due to {e}")

        print("-" * 20)


if __name__ == "__main__":
    main()
