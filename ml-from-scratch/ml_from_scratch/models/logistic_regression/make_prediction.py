import argparse
import logging
import os.path

from ml_from_scratch.constants.data_root_dir import ROOT_DIR
from ml_from_scratch.models.logistic_regression.model_loader import LRModel

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_registry", type=str, default="artefacts", help="Model registry."
    )
    parser.add_argument("--model_dir", type=str, default="lr_model", help="Model dir")
    parser.add_argument("--model_name", type=str, default="lr", help="Model name")
    parser.add_argument(
        "--proba_threshold", type=float, default=0.5, help="Probability threshold"
    )
    args = parser.parse_args()
    return args


def main():
    parsed_args = _parse_args()
    registry_dir_name = parsed_args.model_registry
    model_dir_name = parsed_args.model_dir
    model_artefact_name = parsed_args.model_name
    proba_threshold = parsed_args.proba_threshold

    registry_dir_path = os.path.join(ROOT_DIR, registry_dir_name)
    model_dir_path = os.path.join(registry_dir_path, model_dir_name)
    model_artefact_path = os.path.join(model_dir_path, f"{model_artefact_name}.tar.gz")

    lr_model = LRModel.load_model(model_path=model_artefact_path)

    images_dir = os.path.join(ROOT_DIR, "data/images")
    for img in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img)
        try:
            lr_model.predict_new_image(image_path=img_path, s=proba_threshold)
        except Exception as e:
            logger.error(f"Unable to predict image '{img}' due to {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
