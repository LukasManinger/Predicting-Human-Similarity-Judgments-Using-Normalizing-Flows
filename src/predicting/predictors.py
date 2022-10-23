import json
import math
from collections import namedtuple
from random import shuffle
from typing import Any, Callable, Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from src.glow.glow_additive_network import GlowAdditiveNetwork
from numpy import typing as npt
from tensorflow import keras
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from src import ROOT_DIR
from src.predicting.image_manager import ImageManager

ReferenceSimilarity = namedtuple("ReferenceSimilarity", ["id", "similarity"])


class Predictor:
    def __init__(self, image_manager: ImageManager, dist: Callable, disable_tqdm: bool) -> None:
        self.image_manager = image_manager
        self.dist = dist
        self.disable_tqdm = disable_tqdm

        def preprocess(tensor: tf.Tensor) -> tf.Tensor:
            return tensor

    def predict(self, ranked_observations: npt.NDArray[np.int_]) -> List[ReferenceSimilarity]:
        # ranked_observations = np.delete(ranked_observations, np.where(ranked_observations == 0))

        # prediction = []

        # query = self.image_manager.images[ranked_observations[0]]

        # for reference_id in ranked_observations:
        #     if reference_id == 0:
        #         break

        #     reference = self.image_manager.images[reference_id]

        #     similarity = ReferenceSimilarity(reference_id, self.calc_similarity(query, reference))
        #     prediction.append(similarity)

        # prediction.sort(key=lambda ref_sim: ref_sim.similarity)

        # return prediction

        ranked_observations = np.delete(ranked_observations, np.where(ranked_observations == 0))

        prediction = []

        query = self.latent_images[ranked_observations[0]]

        for i, reference_id in enumerate(ranked_observations):
            reference = self.latent_images[reference_id]

            similarity = ReferenceSimilarity(reference_id, self.calc_similarity(query, reference))
            prediction.append(similarity)

        # shuffle(prediction)
        # prediction.reverse()
        prediction.sort(key=lambda ref_sim: ref_sim.similarity)

        return prediction

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        raise NotImplementedError

    def convert(self) -> None:
        self.latent_images = self.image_manager.images


class SimplePredictor(Predictor):
    def __init__(self, image_manager: ImageManager, dist: Callable, disable_tqdm: bool) -> None:
        super().__init__(image_manager, dist, disable_tqdm)

        def preprocess(tensor: tf.Tensor) -> tf.Tensor:
            tensor = tf.cast(tensor, tf.float32)
            tensor = tf.image.central_crop(tensor, 0.8)
            tensor = tf.image.resize(tensor, (128, 128))

            tensor = tensor / 255 - 0.5  # Should not change anything, just for consistency
            return tensor

        self.image_manager.load_images(preprocess)
        self.convert()

        # self.mse = keras.losses.MeanSquaredError()

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        return self.dist(query, reference).numpy()


class RandomPredictor(Predictor):
    def __init__(self, image_manager: ImageManager, dist: Callable, disable_tqdm: bool) -> None:
        super().__init__(image_manager, dist, disable_tqdm)

        self.rng = np.random.default_rng()

        def preprocess(tensor: tf.Tensor) -> tf.Tensor:
            tensor = tf.cast(tensor, tf.float32)
            tensor = tf.image.central_crop(tensor, 0.8)
            tensor = tf.image.resize(tensor, (128, 128))

            tensor = tensor / 255 - 0.5  # Should not change anything, just for consistency
            return tensor

        self.image_manager.load_images(preprocess)
        self.convert()

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        if tf.reduce_all(query == reference):
            return -1.0

        return self.rng.random()


class GlowPredictor(Predictor):
    def __init__(
        self, image_manager: ImageManager, dist: Callable, hps_path: str, disable_tqdm: bool
    ) -> None:
        super().__init__(image_manager, dist, disable_tqdm)

        self.load_glow(hps_path)

        def preprocess(tensor: tf.Tensor) -> tf.Tensor:
            tensor = tf.cast(tensor, tf.float32)
            tensor = tf.image.central_crop(tensor, self.hps["CENTRAL_CROP_FRACTION"])
            tensor = tf.image.resize(tensor, (self.hps["IMAGE_SIZE"], self.hps["IMAGE_SIZE"]))

            tensor = tensor / 255 - 0.5
            return tensor

        self.image_manager.load_images(preprocess)
        self.convert()

    def load_glow(self, hps_path: str) -> None:
        self.hps: Dict[str, Any]
        with open(hps_path, "r") as json_file:
            self.hps = json.load(json_file)

        coupling = None
        if self.hps["COUPLING_BIJECTOR_FN"] in ["GlowDefaultNetwork", "tfb.GlowDefaultNetwork"]:
            coupling = tfb.GlowDefaultNetwork
        elif self.hps["COUPLING_BIJECTOR_FN"] == "GlowAdditiveNetwork":
            coupling = GlowAdditiveNetwork

        self.glow = tfb.Glow(
            output_shape=self.hps["OUTPUT_SHAPE"],
            num_glow_blocks=self.hps["NUM_GLOW_BLOCKS"],
            num_steps_per_block=self.hps["NUM_STEPS_PER_BLOCK"],
            coupling_bijector_fn=coupling,
            exit_bijector_fn=tfb.GlowDefaultExitNetwork,
            grab_after_block=self.hps["GRAB_AFTER_BLOCK"],
            use_actnorm=self.hps["USE_ACTNORM"],
            seed=self.hps["SEED"],
            validate_args=True,
        )

        self.z_shape = self.glow.inverse_event_shape(self.hps["OUTPUT_SHAPE"])
        self.pz = tfd.Sample(tfd.Normal(0.0, 1.0), self.z_shape)
        self.px = self.glow(self.pz)

        checkpoint = tf.train.Checkpoint(self.glow)

        safe_path = hps_path.replace("hyperparameters.json", "", 1)
        checkpoint_path = f"{safe_path}/checkpoints/ckpt-{math.floor(self.hps['EPOCHS'] / self.hps['CHECKPOINT_FREQ']) + 1}"
        status = checkpoint.restore(checkpoint_path)
        status.assert_consumed().assert_existing_objects_matched().assert_nontrivial_match()

        print(f"Loaded: {checkpoint_path}")

    def convert(self) -> None:
        self.latent_images: Dict[int, tf.Tensor] = dict()

        for id, image in tqdm(
            self.image_manager.images.items(),
            desc="Converting images",
            colour="#7570b3",
            disable=self.disable_tqdm,
        ):
            latent_image = self.glow.inverse(tf.expand_dims(image, 0))
            self.latent_images[id] = latent_image

            if self.disable_tqdm and (id - 1) % 50 == 0:
                print(f"Converting images: {(id - 1)} / {len(self.image_manager.images)}")


class GlowSimplePredictor(GlowPredictor):
    def __init__(
        self, image_manager: ImageManager, dist: Callable, hps_path: str, disable_tqdm: bool
    ) -> None:
        super().__init__(image_manager, dist, hps_path, disable_tqdm)

        # self.mse = keras.losses.MeanSquaredError()

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        return self.dist(query, reference).numpy()


class GlowBayesPredictor(GlowPredictor):
    def __init__(
        self,
        image_manager: ImageManager,
        dist: Callable,
        hps_path: str,
        data_sigma: float,
        disable_tqdm: bool,
    ) -> None:
        super().__init__(image_manager, dist, hps_path, disable_tqdm)

        self.num_dimensions = np.prod(self.hps["OUTPUT_SHAPE"])

        # For computational efficiency we store an list instead of a large diagonal matrix
        self.prior_mean = tf.zeros(self.num_dimensions)
        self.prior_cov = tf.ones(self.num_dimensions)

        self.data_cov = data_sigma * tf.ones(self.num_dimensions)

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        prior_cov_inv = 1 / self.prior_cov
        data_cov_inv = 1 / self.data_cov

        posterior_cov = 1 / (prior_cov_inv + data_cov_inv)

        query_posterior_mean = posterior_cov * (
            prior_cov_inv * self.prior_mean + data_cov_inv * query
        )
        reference_posterior_mean = posterior_cov * (
            prior_cov_inv * self.prior_mean + data_cov_inv * reference
        )

        return self.dist(query_posterior_mean, reference_posterior_mean)
        # return tf.linalg.global_norm([query_posterior_mean - reference_posterior_mean]).numpy()


class GlowWeightedPredictor(GlowPredictor):
    def __init__(
        self, image_manager: ImageManager, dist: Callable, hps_path: str, disable_tqdm: bool
    ) -> None:
        super().__init__(image_manager, dist, hps_path, disable_tqdm)

        ztake = [
            round(
                bs[1] * 4 ** (i + math.log2(self.hps["IMAGE_SIZE"]) - self.hps["NUM_GLOW_BLOCKS"])
            )
            for i, bs in enumerate(self.glow.blockwise_splits)
        ]
        total_z_taken = sum(ztake)
        self.split_sizes = [self.z_shape.as_list()[0] - total_z_taken] + ztake

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        query_zsplits = tf.split(query, num_or_size_splits=self.split_sizes, axis=-1)
        reference_zsplits = tf.split(reference, num_or_size_splits=self.split_sizes, axis=-1)

        distances = []
        for query_split, reference_split in zip(query_zsplits, reference_zsplits):
            if tf.size(query_split).numpy() == 0:
                continue

            distance = self.dist(query_split, reference_split).numpy()
            distances.append(distance)

        return np.mean(distances)


class GlowHalfPredictor(GlowPredictor):
    def __init__(
        self, image_manager: ImageManager, dist: Callable, hps_path: str, disable_tqdm: bool
    ) -> None:
        super().__init__(image_manager, dist, hps_path, disable_tqdm)

        self.mid = round(self.z_shape[0] / 2)
    
    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        return self.dist(query[: self.mid], reference[: self.mid]).numpy()


class PcaPredictor(Predictor):
    def __init__(
        self, image_manager: ImageManager, dist: Callable, emb_name: str, disable_tqdm: bool
    ) -> None:
        super().__init__(image_manager, dist, disable_tqdm)

        self.load_embedding(emb_name)

        def preprocess(tensor: tf.Tensor) -> tf.Tensor:
            tensor = tf.cast(tensor, tf.float32)
            tensor = tf.image.central_crop(tensor, 0.8)
            tensor = tf.image.resize(tensor, (128, 128))

            tensor = tensor / 255 - 0.5
            tensor = tf.reshape(tensor, (-1,))
            return tensor

        self.image_manager.load_images(preprocess)
        self.convert()

    def load_embedding(self, emb_name: str) -> None:
        self.embedding = np.load(
            f"{ROOT_DIR}/pca/safe/{emb_name}.npy", allow_pickle=True, fix_imports=False
        )
        filenames = np.load(
            f"{ROOT_DIR}/pca/safe/{emb_name}_filenames.npy", allow_pickle=True, fix_imports=False
        )

        self.filenames = np.array([filename.split(b"/")[3] for filename in filenames])
        print(self.filenames.shape)

    def convert(self) -> None:
        self.latent_images: Dict[int, tf.Tensor] = dict()

        for id, image in tqdm(
            self.image_manager.images.items(),
            desc="Converting images",
            colour="#7570b3",
            disable=self.disable_tqdm,
        ):
            # Get filename from id
            filepath = self.image_manager.catalog.stimuli.loc[
                self.image_manager.catalog.stimuli["id"] == id
            ]["filepath"].to_list()[0]
            filename = bytes(filepath.split("/")[3], "utf8")
            # Get embedding index from filename
            idx = np.asarray(self.filenames == filename).nonzero()[0][0]

            self.latent_images[id] = self.embedding[idx]

            if self.disable_tqdm and (id - 1) % 50 == 0:
                print(f"Converting images: {(id - 1)} / {len(self.image_manager.images)}")

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        return self.dist(query, reference).numpy()


class ResnetPredictor(Predictor):
    def __init__(
        self,
        image_manager: ImageManager,
        dist: Callable,
        hps_path: str,
        layer_to_use: int,
        disable_tqdm: bool,
    ) -> None:
        super().__init__(image_manager, dist, disable_tqdm)

        self.load_resnet(hps_path, layer_to_use)

        def preprocess(tensor: tf.Tensor) -> tf.Tensor:
            tensor = tf.cast(tensor, tf.float32)
            tensor = keras.applications.resnet50.preprocess_input(tensor)
            tensor = tensor / 255
            # tensor = tf.image.central_crop(tensor, 0.9)  # ???
            tensor = tf.image.resize(tensor, (self.hps["IMAGE_SIZE"], self.hps["IMAGE_SIZE"]))

            return tensor

        self.image_manager.load_images(preprocess)
        self.convert()

    def load_resnet(self, hps_path: str, layer_to_use: int) -> None:
        self.hps: Dict[str, Any]
        with open(hps_path, "r") as json_file:
            self.hps = json.load(json_file)

        safe_path = hps_path.replace("hyperparameters.json", "", 1)
        model_path = f"{safe_path}/model"

        # weights_path = f"{safe_path}/weights/model.tf"
        # print(weights_path)
        # print(self.hps["IMAGE_SIZE"])
        # self.resnet = keras.applications.resnet50.ResNet50(
        #     include_top=True,
        #     weights=None,
        #     input_shape=(self.hps["IMAGE_SIZE"], self.hps["IMAGE_SIZE"], 3),
        #     classes=200,
        # )
        # self.resnet.load_weights(weights_path)

        self.resnet = keras.models.load_model(model_path)

        self.pred_resnet = keras.Model(
            inputs=self.resnet.inputs, outputs=self.resnet.layers[layer_to_use].output
        )

        print(f"{self.resnet.layers[layer_to_use].output.shape = }")
        print(f"Loaded: {self.resnet.name} from {model_path}")

    def convert(self) -> None:
        self.latent_images: Dict[int, tf.Tensor] = dict()

        for id, image in tqdm(
            self.image_manager.images.items(),
            desc="Converting images",
            colour="#7570b3",
            disable=self.disable_tqdm,
        ):
            latent_image = self.pred_resnet(tf.expand_dims(image, 0), training=False)
            self.latent_images[id] = latent_image

            if self.disable_tqdm and (id - 1) % 50 == 0:
                print(f"Converting images: {(id - 1)} / {len(self.image_manager.images)}")

    def calc_similarity(self, query: tf.Tensor, reference: tf.Tensor) -> float:
        return self.dist(query, reference).numpy()  # mean?
        # return tf.linalg.global_norm([query - reference]).numpy()


if __name__ == "__main__":
    # p = MsePredictor(None)

    p = ResnetPredictor(
        None,
        "/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/resnet50/safe/resnet_2022-10-04_12:13:59/hyperparameters.json",
        -2,
        0,
    )
