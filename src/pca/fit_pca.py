import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datetime

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tensorflow import keras

tfb = tfp.bijectors
tfd = tfp.distributions

from src import ROOT_DIR

sns.set_theme("paper", "whitegrid", "Dark2", "DejaVu Sans", rc={"figure.dpi": 100})

now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
print(now)

# HYPER PARAMETERS
HPS = {
    "DATASET": "caltech_birds2011",
    "TRAIN_SLICE": "train+test[:80%]",
    "TEST_SLICE": "test[80%:]",
    "IMAGE_SIZE": 128,
    "CENTRAL_CROP_FRACTION": 0.8,
    "DIR": f"{ROOT_DIR}/pca",
    "TFDS_DATA_DIR": f"{ROOT_DIR}/data",
}


def preprocess(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x["image"], tf.float32)
    x = tf.image.central_crop(x, HPS["CENTRAL_CROP_FRACTION"])
    x = tf.image.resize(x, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))

    x = x / 255 - 0.5
    x = tf.reshape(x, (-1,))
    return x


def depreprocess(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x + 0.5, 0.0, 1.0)


(train_data, full_data), info = tfds.load(
    HPS["DATASET"], split=[HPS["TRAIN_SLICE"], "all"], with_info=True, data_dir=HPS["TFDS_DATA_DIR"]
)

images = train_data.map(preprocess).batch(len(train_data))

x = next(iter(images))

images_full = full_data.map(preprocess).batch(len(full_data))
names_full = full_data.map(lambda t: t["image/filename"]).batch(len(full_data))

x_full = next(iter(images_full))
u_full = next(iter(names_full))

n_components_0 = 10
pca_0 = PCA(n_components=n_components_0)
print("Start fitting 10")
pca_0.fit(x)
print("End fitting")

n_components_1 = 100
pca_1 = PCA(n_components=n_components_1)
print("Start fitting 100")
pca_1.fit(x)
print("End fitting")

n_components_2 = 1000
pca_2 = PCA(n_components=n_components_2)
print("Start fitting 1000")
pca_2.fit(x)
print("End fitting")

z_0 = pca_0.transform(x_full)
z_1 = pca_1.transform(x_full)
z_2 = pca_2.transform(x_full)

comp_0 = pca_0.components_
comp_1 = pca_1.components_
comp_2 = pca_2.components_

print(z_0.shape)
print([np.sum(pca.explained_variance_ratio_) for pca in [pca_0, pca_1, pca_2]])
# -> [0.5287326763682073, 0.7453440771545803, 0.9058766349298837]

n = min(n_components_2, 10)
fig, axs = plt.subplots(1, n, figsize=(7.0866, 1.0), tight_layout=True)

for i in range(n):
    pc = pca_2.components_[i].reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    pc = depreprocess(pc)

    # keras.utils.save_img(f"out/pc_{i}.png", pc * 255, scale=False)

    axs[i].imshow(pc)
    axs[i].set_xticks([])
    axs[i].set_xticklabels([])
    axs[i].set_yticks([])
    axs[i].set_yticklabels([])

plt.savefig(f"{ROOT_DIR}/pca/safe/pca_components.png")
# plt.show()

idxs = [0, 1, 2, 3, 4]

fig, axs = plt.subplots(4, len(idxs), figsize=(7.0866, 4.5), tight_layout=True)

for i, idx in enumerate(idxs):

    orig = depreprocess(x[idx]).numpy().reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    recon_2 = (
        depreprocess(
            np.sum(z_2[idx].reshape((n_components_2, -1)) * comp_2, axis=0).reshape(
                (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3)
            )
        )
        .numpy()
        .reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    )
    recon_1 = (
        depreprocess(
            np.sum(z_1[idx].reshape((n_components_1, -1)) * comp_1, axis=0).reshape(
                (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3)
            )
        )
        .numpy()
        .reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    )
    recon_0 = (
        depreprocess(
            np.sum(z_0[idx].reshape((n_components_0, -1)) * comp_0, axis=0).reshape(
                (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3)
            )
        )
        .numpy()
        .reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    )

    # keras.utils.save_img(f"out/0{i}.png", orig * 255, scale=False)
    # keras.utils.save_img(f"out/1{i}.png", recon_2 * 255, scale=False)
    # keras.utils.save_img(f"out/2{i}.png", recon_1 * 255, scale=False)
    # keras.utils.save_img(f"out/3{i}.png", recon_0 * 255, scale=False)

    axs[0, i].imshow(orig)
    axs[0, i].set_xticks([])
    axs[0, i].set_xticklabels([])
    axs[0, i].set_yticks([])
    axs[0, i].set_yticklabels([])

    axs[1, i].imshow(recon_2)
    axs[1, i].set_xticks([])
    axs[1, i].set_xticklabels([])
    axs[1, i].set_yticks([])
    axs[1, i].set_yticklabels([])

    axs[2, i].imshow(recon_1)
    axs[2, i].set_xticks([])
    axs[2, i].set_xticklabels([])
    axs[2, i].set_yticks([])
    axs[2, i].set_yticklabels([])

    axs[3, i].imshow(recon_0)
    axs[3, i].set_xticks([])
    axs[3, i].set_xticklabels([])
    axs[3, i].set_yticks([])
    axs[3, i].set_yticklabels([])

axs[0, 0].set_ylabel("Original")
axs[1, 0].set_ylabel("Reconstruction\nPCA (1000)")
axs[2, 0].set_ylabel("Reconstruction\nPCA (100)")
axs[3, 0].set_ylabel("Reconstruction\nPCA (10)")

plt.savefig(f"{ROOT_DIR}/pca/safe/reconstruction.png")
# plt.show()

np.save(
    f"{ROOT_DIR}/pca/safe/pca_emb_{n_components_0}.npy", z_0, allow_pickle=True, fix_imports=False
)
np.save(
    f"{ROOT_DIR}/pca/safe/pca_emb_{n_components_0}_filenames.npy",
    u_full.numpy(),
    allow_pickle=True,
    fix_imports=False,
)

np.save(
    f"{ROOT_DIR}/pca/safe/pca_emb_{n_components_1}.npy", z_1, allow_pickle=True, fix_imports=False
)
np.save(
    f"{ROOT_DIR}/pca/safe/pca_emb_{n_components_1}_filenames.npy",
    u_full.numpy(),
    allow_pickle=True,
    fix_imports=False,
)

np.save(
    f"{ROOT_DIR}/pca/safe/pca_emb_{n_components_2}.npy", z_2, allow_pickle=True, fix_imports=False
)
np.save(
    f"{ROOT_DIR}/pca/safe/pca_emb_{n_components_2}_filenames.npy",
    u_full.numpy(),
    allow_pickle=True,
    fix_imports=False,
)
