from typing import Callable, Dict

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import typing as npt
from psiz.catalog.catalog import Catalog
from tensorflow import keras
from tqdm import tqdm

from src import ROOT_DIR


class ImageManager:
    def __init__(self, catalog: Catalog, disable_tqdm: bool) -> None:
        self.catalog = catalog
        self.disable_tqdm = disable_tqdm

        # self.load_images()

    def load_images(
        self,
        preprocess: Callable[[tf.Tensor], tf.Tensor],
    ) -> None:
        self.images: Dict[int, tf.Tensor] = dict()

        for stimulus in tqdm(
            self.catalog.stimuli.itertuples(index=False, name="Stimulus"),
            desc="Loading images",
            total=len(self.catalog.stimuli.index),
            colour="#1b9e77",
            disable=self.disable_tqdm,
        ):
            image = self.load_image(f"{ROOT_DIR}/data/{stimulus.filepath}")
            image = preprocess(image)
            self.images[stimulus.id] = image

            if self.disable_tqdm and (stimulus.id - 1) % 50 == 0:
                print(f"Loading images: {(stimulus.id - 1)} / {len(self.catalog.stimuli.index)}")

    def load_image(self, path: str) -> tf.Tensor:
        return tf.convert_to_tensor(keras.utils.img_to_array(keras.utils.load_img(path)))

    def visualize_stimulus(self, stimulus: npt.NDArray[np.int_]) -> None:
        fig, axs = plt.subplots(3, 3, tight_layout=True)

        for img_id, ax in zip(stimulus, axs.flat):
            ax.imshow(self.images[img_id].numpy() / 255.0)
            ax.set_axis_off()
            ax.set_title(img_id)

        plt.show()
