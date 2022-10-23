import os
import time
import datetime
import json
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import scipy as sp
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

#!pip install umap-learn
# import umap

# plt.rcParams["figure.dpi"] = 100
# plt.rcParams["figure.figsize"] = (10, 5)
# plt.style.use("seaborn-darkgrid")

sns.set_theme("paper", "whitegrid", "Dark2", "DejaVu Sans", rc={"figure.dpi": 100})
# print(plt.rcParams)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
print(now)

# HYPER PARAMETERS
HPS = {
    "DATASET": "caltech_birds2011",
    "TRAIN_SLICE": "train+test[:80%]",  # "train+test[:80%]",  # Use shards?
    "TEST_SLICE": "test[80%:]",
    "IMAGE_SIZE": 128,
    "CENTRAL_CROP_FRACTION": 0.8,
    "NOTEBOOK_DIR": "/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/evaluation/embeddings",
    "TFDS_DATA_DIR": "/work/scratch/lm83qyjo/tfds_data_dir",
}

# HPS["TFDS_DATA_DIR"] = f"{HPS['NOTEBOOK_DIR']}/tfds_data_dir"


def preprocess(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x["image"], tf.float32)
    x = tf.image.central_crop(x, HPS["CENTRAL_CROP_FRACTION"])
    x = tf.image.resize(x, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))

    x = x / 255 - 0.5
    x = tf.reshape(x, (-1,))
    return x


def depreprocess(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x + 0.5, 0.0, 1.0)
    # return x


(train_data, full_data), info = tfds.load(
    HPS["DATASET"], split=[HPS["TRAIN_SLICE"], "all"], with_info=True, data_dir=HPS["TFDS_DATA_DIR"]
)
# train_data, info = tfds.load(HPS["DATASET"], split=HPS["TRAIN_SLICE"], with_info=True, data_dir=HPS["TFDS_DATA_DIR"])
# print(info)

images = train_data.map(preprocess).batch(len(train_data))
# labels = train_data.map(lambda t: t["label"]).batch(len(train_data))
# names = train_data.map(lambda t: t["image/filename"]).batch(len(train_data))

x = next(iter(images))
# y = next(iter(labels))
# u = next(iter(names))

images_full = full_data.map(preprocess).batch(len(full_data))
# labels_full = full_data.map(lambda t: t["label"]).batch(len(full_data))
names_full = full_data.map(lambda t: t["image/filename"]).batch(len(full_data))

x_full = next(iter(images_full))
# y_full = next(iter(labels_full))
u_full = next(iter(names_full))

# del train_data
# del full_data

# train_data, info = tfds.load("mnist", split="train", with_info=True, data_dir=HPS["TFDS_DATA_DIR"])
# print(info)

# train_images = train_data.map(lambda t: tf.reshape(tf.cast(t["image"], tf.float32) / 255 - 0.5, (-1,))).batch(len(train_data))
# train_labels = train_data.map(lambda t: t["label"]).batch(len(train_data))

# x = next(iter(train_images))
# y = next(iter(train_labels))

# print(type(x))
# print(x.shape)

# print()

# print(type(x_full))
# print(x_full.shape)

# print()

# #print(type(y_full))
# #print(y_full.shape)
# #print(y[:3])

# print()

# print(type(u_full))
# print(u_full.shape)
# #print(u[:3])

# d1 = x[0]

# d2 = tf.reshape(d1, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
# d3 = tf.reshape(d2, (-1,))
# d4 = tf.reshape(d3, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))

# print(tf.reduce_all(d1 == d3).numpy())
# print(tf.reduce_all(d2 == d4).numpy())

n_components_0 = 10
pca_0 = PCA(n_components=n_components_0)
print("Start fitting")
pca_0.fit(x)
print("End fitting")

n_components_1 = 100
pca_1 = PCA(n_components=n_components_1)
print("Start fitting")
pca_1.fit(x)
print("End fitting")

n_components_2 = 1000
pca_2 = PCA(n_components=n_components_2)
print("Start fitting")
pca_2.fit(x)
print("End fitting")

if False:
    with open("pca_222.pkl", "wb") as file:
        pickle.dump(pca, file)

if False:
    with open("pca_222.pkl", "rb") as file:
        pca = pickle.load(file)
        print(type(pca))

z_0 = pca_0.transform(x_full)
z_1 = pca_1.transform(x_full)
z_2 = pca_2.transform(x_full)

comp_0 = pca_0.components_
comp_1 = pca_1.components_
comp_2 = pca_2.components_

print(z_0.shape)
# print(comp.shape)
# print(pca.explained_variance_[:10])
# print(pca.explained_variance_ratio_[:10])

print([np.sum(pca.explained_variance_ratio_) for pca in [pca_0, pca_1, pca_2]])
# -> [0.5287326763682073, 0.7453440771545803, 0.9058766349298837]

# fig, axs = plt.subplots(1, 2, constrained_layout=True)

# axs[0].scatter(x_full[:, 0], x_full[:, 1], fc="none", ec="black", alpha=0.1)
# axs[1].scatter(z[:, 0], z[:, 1], fc="none", ec="black", alpha=0.1)

# plt.show()

n = min(n_components_2, 10)
fig, axs = plt.subplots(1, n, figsize=(7.0866, 1.0), tight_layout=True)

for i in range(n):
    pc = pca_2.components_[i].reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    pc = depreprocess(pc)

    keras.utils.save_img(f"out/pc_{i}.png", pc * 255, scale=False)

    #print(f"pc {i}: {np.min(pc)}, {np.max(pc)}")
    axs[i].imshow(pc)
    # axs[i].set_axis_off()
    axs[i].set_xticks([])
    axs[i].set_xticklabels([])
    axs[i].set_yticks([])
    axs[i].set_yticklabels([])

# plt.show()
plt.savefig("pca_components.png")
plt.savefig("pca_components.pdf")

idxs = [0, 1, 2, 3, 4]

fig, axs = plt.subplots(4, len(idxs), figsize=(7.0866, 4.5), tight_layout=True)

for i, idx in enumerate(idxs):

    orig = depreprocess(x[idx]).numpy().reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    recon_2 = depreprocess(
        np.sum(z_2[idx].reshape((n_components_2, -1)) * comp_2, axis=0).reshape(
            (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3)
        )
    ).numpy().reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    recon_1 = depreprocess(
        np.sum(z_1[idx].reshape((n_components_1, -1)) * comp_1, axis=0).reshape(
            (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3)
        )
    ).numpy().reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))
    recon_0 = depreprocess(
        np.sum(z_0[idx].reshape((n_components_0, -1)) * comp_0, axis=0).reshape(
            (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3)
        )
    ).numpy().reshape((HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3))

    keras.utils.save_img(f"out/0{i}.png", orig * 255, scale=False)
    keras.utils.save_img(f"out/1{i}.png", recon_2 * 255, scale=False)
    keras.utils.save_img(f"out/2{i}.png", recon_1 * 255, scale=False)
    keras.utils.save_img(f"out/3{i}.png", recon_0 * 255, scale=False)


    # print(f"orig {i}: {np.min(orig)}, {np.max(orig)}")
    axs[0, i].imshow(orig)
    # axs[0, i].set_axis_off()
    axs[0, i].set_xticks([])
    axs[0, i].set_xticklabels([])
    axs[0, i].set_yticks([])
    axs[0, i].set_yticklabels([])

    # print(f"recon {i}: {np.min(recon)}, {np.max(recon)}")
    axs[1, i].imshow(recon_2)
    # axs[1, i].set_axis_off()
    axs[1, i].set_xticks([])
    axs[1, i].set_xticklabels([])
    axs[1, i].set_yticks([])
    axs[1, i].set_yticklabels([])

    axs[2, i].imshow(recon_1)
    # axs[2, i].set_axis_off()
    axs[2, i].set_xticks([])
    axs[2, i].set_xticklabels([])
    axs[2, i].set_yticks([])
    axs[2, i].set_yticklabels([])

    axs[3, i].imshow(recon_0)
    # axs[2, i].set_axis_off()
    axs[3, i].set_xticks([])
    axs[3, i].set_xticklabels([])
    axs[3, i].set_yticks([])
    axs[3, i].set_yticklabels([])

axs[0, 0].set_ylabel("Original")
axs[1, 0].set_ylabel("Reconstruction\nPCA (1000)")
axs[2, 0].set_ylabel("Reconstruction\nPCA (100)")
axs[3, 0].set_ylabel("Reconstruction\nPCA (10)")

# plt.show()
plt.savefig("reconstruction2.png")
plt.savefig("reconstruction2.pdf")

if False:
    np.save(f"pca_emb_{n_components}.npy", z, allow_pickle=True, fix_imports=False)
    np.save(
        f"pca_emb_{n_components}_filenames.npy",
        u_full.numpy(),
        allow_pickle=True,
        fix_imports=False,
    )

if True:
    np.save(f"pca_emb_{n_components_0}.npy", z_0, allow_pickle=True, fix_imports=False)
    np.save(
        f"pca_emb_{n_components_0}_filenames.npy",
        u_full.numpy(),
        allow_pickle=True,
        fix_imports=False,
    )
    
    np.save(f"pca_emb_{n_components_1}.npy", z_1, allow_pickle=True, fix_imports=False)
    np.save(
        f"pca_emb_{n_components_1}_filenames.npy",
        u_full.numpy(),
        allow_pickle=True,
        fix_imports=False,
    )
    
    np.save(f"pca_emb_{n_components_2}.npy", z_2, allow_pickle=True, fix_imports=False)
    np.save(
        f"pca_emb_{n_components_2}_filenames.npy",
        u_full.numpy(),
        allow_pickle=True,
        fix_imports=False,
    )

# z_tsne = TSNE(n_components=2, learning_rate="auto", init="pca", verbose=1).fit_transform(z)

# fig, ax = plt.subplots(1, 1, constrained_layout=True)

# ax.scatter(z_tsne[:, 0], z_tsne[:, 1], fc="none", ec=[f"C{val}" for val in y], alpha=0.1)

# plt.show()

##z_umap = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.2, spread=2.0).fit_transform(z)
# z_umap = umap.UMAP(n_components=2).fit_transform(z)

# z_full_umap = umap.UMAP(n_components=2).fit_transform(x)

# fig, axs = plt.subplots(1, 2, constrained_layout=True)

# axs[0].scatter(z_umap[:, 0], z_umap[:, 1], c=y, cmap="hsv", alpha=0.1)
# axs[1].scatter(z_full_umap[:, 0], z_full_umap[:, 1], c=y, cmap="hsv", alpha=0.1)

# plt.show()

# if input("SAVE? ").lower() == "y":
if False:
    np.save("test.npy", z, fix_imports=False)

# nu = u_full.numpy()
# print(nu)

# for name in nu:
#     nn = name.split(b"/")[3]
#     if nn.startswith(b"Blue_Grosbeak_0002"):
#         print(nn)
