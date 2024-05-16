import logging
from typing import Tuple
import h5py
import numpy as np
from data_types import TidyData

logger = logging.getLogger(__name__)


class DataPreparation:

    @staticmethod
    def h5_train_loader(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        train_dataset = h5py.File(name=data_path, mode="r")
        train_x = np.array(train_dataset["train_set_x"][:])
        train_y = np.array(train_dataset["train_set_y"][:])
        train_y = train_y.reshape((1, train_y.shape[0]))
        return train_x, train_y

    @staticmethod
    def h5_test_loader(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        test_dataset = h5py.File(name=data_path, mode="r")
        test_x = np.array(test_dataset["test_set_x"][:])
        test_y = np.array(test_dataset["test_set_y"][:])
        test_y = test_y.reshape((1, test_y.shape[0]))
        classes = np.array(test_dataset["list_classes"][:])
        return test_x, test_y, classes

    @staticmethod
    def data_preprocessing(data: np.ndarray) -> np.ndarray:
        """
        * Reshape images of size (number_pixels, number_pixels, 3) into a vectors
        of shape (number_pixels ∗ number_pixels ∗ 3, 1)


        * Standardize training dataset
        To represent color images, the red, green and blue channels (RGB) must be specified for each pixel,
        and so the pixel value is actually a vector of three numbers ranging from 0 to 255.
        One common preprocessing step in machine learning is to center and standardize your dataset,
        meaning that you subtract the mean of the whole numpy array from each example,
        and then divide each example by the standard deviation of the whole numpy array.
        But for picture datasets, it is simpler and more convenient and works almost as well
        to just divide every row of the dataset by 255 (the maximum value of a pixel channel).

            *  Apply this transformation: (X - Mean(X))/STD(X) where X is a feature vector
                * In this context of images, just divide every row of the dataset by 255

        :return:
        """

        logger.info(f"Before flattening data: initial shape =>  {data.shape}")

        num_samples = data.shape[0]
        data_flatten = data.reshape(num_samples, -1).T
        logger.info(f"Flattened data is a numpy-array where each column represents a flattened image: "
                    f"new shape => {data_flatten.shape}")

        data_normalized = data_flatten / 255
        return data_normalized

    def load_data(self, train_path: str, test_path: str):
        train_x, train_y = self.h5_train_loader(data_path=train_path)
        test_x, test_y, classes = self.h5_test_loader(data_path=test_path)

        logger.debug(f"Train X.shape: {train_x.shape}")
        logger.debug(f"Train Y.shape: {train_y.shape}")
        logger.debug(f"Test X.shape: {test_x.shape}")
        logger.debug(f"Test Y.shape: {test_y.shape}")

        logger.info(f"Number of training examples: {train_x.shape[0]}")
        logger.info(f"Number of testing examples: {test_x.shape[0]}")
        logger.info(f"Height/Width of each image: {train_x.shape[1]}")
        logger.info(f"Each image is of size: ({train_x.shape[1]}, {train_x.shape[1]}, 3)")

        train_x_normalized = self.data_preprocessing(data=train_x)
        test_x_normalized = self.data_preprocessing(data=test_x)

        logger.info(f"Train shape after preprocessing: {train_x_normalized.shape}")
        logger.info(f"Test shape after preprocessing: {test_x_normalized.shape}")

        res = TidyData(train_x_normalized, train_y, test_x_normalized, test_y, classes)

        return res


