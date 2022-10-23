# %%
import json
import math
from collections import namedtuple
from typing import Any, Dict, List

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm, trange
from numpy import typing as npt
from tensorflow import keras
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

# %%
def load_glow(hps_path: str) -> None:
    hps: Dict[str, Any]
    with open(hps_path, "r") as json_file:
        hps = json.load(json_file)

    glow = tfb.Glow(
        output_shape=hps["OUTPUT_SHAPE"],
        num_glow_blocks=hps["NUM_GLOW_BLOCKS"],
        num_steps_per_block=hps["NUM_STEPS_PER_BLOCK"],
        coupling_bijector_fn=tfb.GlowDefaultNetwork,
        exit_bijector_fn=tfb.GlowDefaultExitNetwork,
        grab_after_block=hps["GRAB_AFTER_BLOCK"],
        use_actnorm=hps["USE_ACTNORM"],
        seed=hps["SEED"],
        validate_args=True,
    )

    z_shape = glow.inverse_event_shape(hps["OUTPUT_SHAPE"])
    pz = tfd.Sample(tfd.Normal(0.0, 1.0), z_shape)
    px = glow(pz)

    checkpoint = tf.train.Checkpoint(glow)

    checkpoint_path = f"{hps['CHECKPOINT_DIR']}-{math.floor(hps['EPOCHS'] / hps['CHECKPOINT_FREQ']) + 1}"
    status = checkpoint.restore(checkpoint_path)
    status.assert_consumed().assert_existing_objects_matched().assert_nontrivial_match()

    print(f"Loaded: {checkpoint_path}")

    return glow, pz, px, hps

# %%
#glow, pz, px, hps = load_glow("/home/lukas/HESSENBOX-DA/Bachelor-Thesis/Ergebnisse/TFP GLOW/64-1_2022-09-05_05:59:46/hyperparameters.json")
glow, pz, px, hps = load_glow("/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/safe/128-2_2022-09-20_17:36:27/hyperparameters.json")

# %%
def load_image(path: str) -> tf.Tensor:
    return tf.convert_to_tensor(keras.utils.img_to_array(keras.utils.load_img(path)))

# %%
def preprocess(tensor: tf.Tensor) -> tf.Tensor:
    tensor = tf.cast(tensor, tf.float32)
    tensor = tf.image.central_crop(tensor, hps["CENTRAL_CROP_FRACTION"])
    tensor = tf.image.resize(tensor, (hps["IMAGE_SIZE"], hps["IMAGE_SIZE"]))

    tensor = tensor / 255 - 0.5
    return tensor

# %%
def depreprocess(tensor: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(tensor + 0.5, 0.0, 1.0)

# %%
img1 = load_image("/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/evaluation/images/Bird/Cardinalidae/Blue_Grosbeak/Blue_Grosbeak_0002_36648.jpg") 
img1 = preprocess(img1)

img2 = load_image("/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/evaluation/images/Bird/Cardinalidae/Blue_Grosbeak/Blue_Grosbeak_0014_36708.jpg")
img2 = preprocess(img2) 

img3 = load_image("/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/evaluation/images/Bird/Icteridae/Scott_Oriole/Scott_Oriole_0010_795852.jpg")
img3 = preprocess(img3)

img4 = load_image("/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/evaluation/images/Bird/Icteridae/Bobolink/Bobolink_0049_9540.jpg")
img4 = preprocess(img4)

# %%
fig, axs = plt.subplots(1, 4, dpi=100, constrained_layout=True)

for img, ax in zip([img1, img2, img3, img4], axs):
    ax.imshow(depreprocess(img))
    ax.set_axis_off()

plt.show()

# %%
def interpolate(img1, img2, n):
    interpolations = []

    z1, z2 = glow.inverse(tf.stack((img1, img2)))

    vec = z2 - z1

    for i in trange(n):
        t = z1 + (i / (n - 1)) * vec
        t = glow(tf.expand_dims(t, 0))
        interpolations.append(t)
    
    return interpolations

# %%
def interpolate_pixel(img1, img2, n):
    interpolations = []

    vec = img2 - img1

    for i in trange(n):
        t = img1 + (i / (n - 1)) * vec
        interpolations.append(t)
    
    return interpolations

# %%
n = 5
interpolations = interpolate(img1, img2, n)

fig, axs = plt.subplots(1, n, dpi=256, constrained_layout=True)

for i in range(n):
    keras.utils.save_img(f"out/interpolate/a{i}.png", depreprocess(interpolations[i][0]).numpy() * 255, scale=False)
    axs[i].imshow(depreprocess(interpolations[i][0]))
    axs[i].set_axis_off()

plt.savefig("out/interpolate/1->2.png")
plt.show()

# %%
n = 5
interpolations = interpolate(img2, img3, n)

fig, axs = plt.subplots(1, n, dpi=256, constrained_layout=True)

for i in range(n):
    keras.utils.save_img(f"out/interpolate/b{i}.png", depreprocess(interpolations[i][0]).numpy() * 255, scale=False)
    axs[i].imshow(depreprocess(interpolations[i][0]))
    axs[i].set_axis_off()

plt.savefig("out/interpolate/2->3.png")
plt.show()

# %%
n = 5
interpolations = interpolate(img3, img4, n)

fig, axs = plt.subplots(1, n, dpi=256, constrained_layout=True)

for i in range(n):
    keras.utils.save_img(f"out/interpolate/c{i}.png", depreprocess(interpolations[i][0]).numpy() * 255, scale=False)
    axs[i].imshow(depreprocess(interpolations[i][0]))
    axs[i].set_axis_off()

plt.savefig("out/interpolate/3->4.png")
plt.show()

# %%
# n = 7
# interpolations = interpolate_pixel(img1, img2, n)

# fig, axs = plt.subplots(1, n, dpi=256, constrained_layout=True)

# for i in range(n):
#     axs[i].imshow(depreprocess(interpolations[i]))
#     axs[i].set_axis_off()

# plt.show()

# %%
# n = 7
# interpolations = interpolate_pixel(img2, img3, n)

# fig, axs = plt.subplots(1, n, dpi=256, constrained_layout=True)

# for i in range(n):
#     axs[i].imshow(depreprocess(interpolations[i]))
#     axs[i].set_axis_off()

# plt.show()

# %%
# n = 7
# interpolations = interpolate_pixel(img3, img4, n)

# fig, axs = plt.subplots(1, n, dpi=256, constrained_layout=True)

# for i in range(n):
#     axs[i].imshow(depreprocess(interpolations[i]))
#     axs[i].set_axis_off()

# plt.show()

# %%



