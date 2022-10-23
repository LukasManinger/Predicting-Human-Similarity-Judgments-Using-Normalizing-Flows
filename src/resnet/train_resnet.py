import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import datetime
import json

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow import keras

from src import ROOT_DIR

sns.set_theme("paper", "whitegrid", "Dark2", "SejaVu Sans", rc={"figure.dpi": 100})


def preprocess(x: tf.Tensor, training: bool) -> tf.Tensor:
    if HPS["PREPROCESS_MODE"] == 1:
        t = tf.cast(x["image"], tf.float32)
        t = keras.applications.resnet50.preprocess_input(t)
        t = t / 255

        t = tf.image.resize(t, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))

        if training:
            data_augmentation = tf.keras.Sequential(
                [
                    keras.layers.RandomFlip("horizontal_and_vertical"),
                    keras.layers.RandomContrast(0.1),
                    keras.layers.RandomZoom(0.1),
                    keras.layers.RandomRotation(0.1),
                    keras.layers.GaussianNoise(0.02),
                ]
            )

            t = data_augmentation(t, training=True)

    elif HPS["PREPROCESS_MODE"] == 2:
        t = tf.cast(x["image"], tf.float32)
        t = keras.applications.resnet50.preprocess_input(t)
        t = t / 255

        t = tf.image.resize(t, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))

        if training:
            data_augmentation = tf.keras.Sequential(
                [
                    keras.layers.RandomFlip("horizontal_and_vertical"),
                    keras.layers.RandomContrast(0.2),
                    keras.layers.RandomRotation(0.1),
                    keras.layers.RandomTranslation(0.1, 0.1),
                    keras.layers.RandomZoom(0.2),
                    keras.layers.GaussianNoise(0.04),
                ]
            )

            t = data_augmentation(t, training=True)

    return t, x["label"]


now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(f"{now = }")

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--preprocess_mode", default=2, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
# parser.add_argument("--optimizer_global_clipnorm", default=1.0, type=float)
parser.add_argument("--cosine_decay_steps", default=1000, type=int)
parser.add_argument("--cosine_decay_t_mul", default=2.0, type=float)
parser.add_argument("--cosine_decay_m_mul", default=1.0, type=float)
parser.add_argument("--cosine_decay_alpha", default=0.0, type=float)
parser.add_argument("--fit_verbose", default=1, type=int)
parser.add_argument("--name", default="resnet", type=str)

cmd_args = parser.parse_args()
print(f"{cmd_args = }")

HPS = {
    "DATASET": "caltech_birds2011",
    "TRAIN_SLICE": "train+test[:80%]",
    "TEST_SLICE": "test[80%:]",
    "IMAGE_SIZE": 128,
    "FIT_VERBOSE": cmd_args.fit_verbose,
    "PREPROCESS_MODE": cmd_args.preprocess_mode,
    # "CENTRAL_CROP_FRACTION": 0.8,
    "SEED": cmd_args.seed,
    "EPOCHS": cmd_args.epochs,
    "BATCH_SIZE": cmd_args.batch_size,
    "LEARNING_RATE": cmd_args.learning_rate,
    # "OPTIMIZER_GLOBAL_CLIPNORM": cmd_args.optimizer_global_clipnorm,
    "COSINE_DECAY_STEPS": cmd_args.cosine_decay_steps,
    "COSINE_DECAY_T_MUL": cmd_args.cosine_decay_t_mul,
    "COSINE_DECAY_M_MUL": cmd_args.cosine_decay_m_mul,
    "COSINE_DECAY_ALPHA": cmd_args.cosine_decay_alpha,
    "ID": ...,
    "DIR": f"{ROOT_DIR}/resnet",
    "SAFE_DIR": ...,
    "TFDS_DATA_DIR": f"{ROOT_DIR}/data",
}

HPS["ID"] = f"{cmd_args.name}_{now}"
HPS["SAFE_DIR"] = f"{HPS['DIR']}/safe/{HPS['ID']}"

os.makedirs(HPS["SAFE_DIR"], exist_ok=True)

with open(f"{HPS['SAFE_DIR']}/hyperparameters.json", "w") as json_file:
    json.dump(HPS, json_file, indent=4)

tf.random.set_seed(HPS["SEED"])

(train_data, test_data), info = tfds.load(
    HPS["DATASET"],
    split=[HPS["TRAIN_SLICE"], HPS["TEST_SLICE"]],
    with_info=True,
    data_dir=HPS["TFDS_DATA_DIR"],
)
# as_supervised=True?
# print(info)

train_data = train_data.shuffle(len(train_data), reshuffle_each_iteration=True)

train_images = train_data.map(lambda x: preprocess(x, True)).batch(HPS["BATCH_SIZE"])
# train_labels = train_data.map(lambda t: t["label"]).batch(HPS["BATCH_SIZE"])

test_images = test_data.map(lambda x: preprocess(x, False)).batch(HPS["BATCH_SIZE"])
# test_labels = test_data.map(lambda t: t["label"]).batch(HPS["BATCH_SIZE"])

# MODEL

model = keras.applications.resnet50.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3),
    # pooling="avg",
    classes=200,
)
# model.summary()

# LEARNING

# class PrintLearningRate(Callback):
#     def __init__(self):
#         pass

#     def on_epoch_begin(self, epoch, logs=None):
#         lr = K.eval(self.model.optimizer._decayed_lr(tf.float64)
#         print("\nLearning rate at epoch {} is {}".format(epoch, lr)))

lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    HPS["LEARNING_RATE"],
    HPS["COSINE_DECAY_STEPS"],
    HPS["COSINE_DECAY_T_MUL"],
    HPS["COSINE_DECAY_M_MUL"],
    HPS["COSINE_DECAY_ALPHA"],
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    train_images, epochs=HPS["EPOCHS"], validation_data=test_images, verbose=HPS["FIT_VERBOSE"]
)

# print(model.optimizer.lr.get_config())
# print(f"{model.optimizer.learning_rate = }")

# PLOTTING

fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)

axs[0].plot(history.history["loss"], label="loss")
axs[0].plot(history.history["val_loss"], label="val_loss")
# axs[0].set_xlabel('Epoch')
axs[0].set_ylabel("Crossentropy")
axs[0].legend()

axs[1].plot(history.history["accuracy"], label="accuracy")
axs[1].plot(history.history["val_accuracy"], label="val_accuracy")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()

plt.savefig(f"{HPS['SAFE_DIR']}/plot.png")

test_loss, test_acc = model.evaluate(test_images, verbose=2)

print(f"{test_loss = }")
print(f"{test_acc = }")

# SAVE MODEL

model.save(f"{HPS['SAFE_DIR']}/model")

history_df = pd.DataFrame(history.history)
with open(f"{HPS['SAFE_DIR']}/history.csv", mode="w") as csv_file:
    history_df.to_csv(csv_file)
