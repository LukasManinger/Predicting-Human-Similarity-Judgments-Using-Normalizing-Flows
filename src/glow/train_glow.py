import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import datetime
import json
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import seaborn as sns
from matplotlib import pyplot as plt
from src import ROOT_DIR
from src.glow.glow_additive_network import GlowAdditiveNetwork
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tqdm import tqdm, trange

sns.set_theme("paper", "whitegrid", "Dark2", "SejaVu Sans", rc={"figure.dpi": 100})

def arg_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="caltech_birds2011", type=str)
    parser.add_argument("--train_slice", default="train+test[:80%]", type=str)
    parser.add_argument("--test_slice", default="test[80%:]", type=str)
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--central_crop_fraction", default=0.8, type=float)
    parser.add_argument("--preprocess_mode", default=3, type=int)
    parser.add_argument("--n_bits", default=5, type=int)
    parser.add_argument("--not_disable_tqdm", action="store_false")
    parser.add_argument("--num_glow_blocks", default=3, type=int)
    parser.add_argument("--num_steps_per_block", default=32, type=int)
    parser.add_argument("--coupling_bijector_fn", default="GlowDefaultNetwork", type=str)
    parser.add_argument("--grab_after_block", default=None, type=tuple)
    parser.add_argument("--not_use_actnorm", action="store_false")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--optimizer_clipvalue", default=10.0, type=float)
    parser.add_argument("--optimizer_global_clipnorm", default=1.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-7, type=float)
    parser.add_argument("--cosine_decay_steps", default=10, type=int)
    parser.add_argument("--cosine_decay_t_mul", default=2.0, type=float)
    parser.add_argument("--cosine_decay_m_mul", default=1.0, type=float)
    parser.add_argument("--cosine_decay_alpha", default=0.0, type=float)
    parser.add_argument("--checkpoint_freq", default=10, type=int)
    parser.add_argument("--name", default="tfp_glow", type=str)

    return parser.parse_args()


def preprocess(x: tf.Tensor) -> tf.Tensor:
    # TODO reduce bits?
    if HPS["PREPROCESS_MODE"] == 0:
        print("Warning: PREPROCESS_MODE == 0 is deprecated")
        # Cast to float
        x = tf.cast(x["image"], tf.float32)
        # Resize to target size (does not preserve aspect ratio)
        x = tf.image.resize(x, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))
        # Random horizontal flip (50 % chance)
        x = tf.image.random_flip_left_right(x)
        # Add noise and scale into [0, 1]
        x = (x + tf.random.uniform(x[0].shape)) / 256
    elif HPS["PREPROCESS_MODE"] == 1:
        print("Warning: PREPROCESS_MODE == 1 is deprecated")
        # Cast to float
        x = tf.cast(x["image"], tf.float32)
        # Resize to target size (does not preserve aspect ratio)
        x = tf.image.resize(x, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))
        # Random horizontal flip (50 % chance)
        x = tf.image.random_flip_left_right(x)
        # Add noise and scale into [-1, +1]
        x = ((x + tf.random.uniform(x[0].shape)) / 128) - 1
    elif HPS["PREPROCESS_MODE"] == 2:
        print("Warning: PREPROCESS_MODE == 2 is deprecated")
        # Cast to float
        x = tf.cast(x["image"], tf.float32)
        # Crop the outer parts of the image
        x = tf.image.central_crop(x, HPS["CENTRAL_CROP_FRACTION"])
        # Resize to target size (does not preserve aspect ratio)
        x = tf.image.resize(x, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))
        # Random horizontal flip (50 % chance)
        x = tf.image.random_flip_left_right(x)
        # Add noise and scale into [-2, +2]
        x = ((x + tf.random.uniform(x[0].shape)) / 64) - 2
    elif HPS["PREPROCESS_MODE"] == 3:
        # Cast to float
        x = tf.cast(x["image"], tf.float32)
        # Crop the outer parts of the image
        x = tf.image.central_crop(x, HPS["CENTRAL_CROP_FRACTION"])
        # Resize to target size (does not preserve aspect ratio)
        x = tf.image.resize(x, (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"]))
        # Random horizontal flip (50 % chance)
        x = tf.image.random_flip_left_right(x)

        # Reduce bits
        if HPS["N_BITS"] < 8:
            x = tf.floor(x / 2 ** (8 - HPS["N_BITS"]))

        # Scale into [-0.5, +0.5]
        x = x / HPS["N_BINS"] - 0.5
    else:
        print("Invalid PREPROCESS_MODE!")

    return x


def depreprocess(x: tf.Tensor) -> tf.Tensor:
    # Not 100 % inverse to 'preprocess'
    if HPS["PREPROCESS_MODE"] == 0:
        # | 0, 1 | --> [0, 1]
        return tf.clip_by_value(x, 0.0, 1.0)
    elif HPS["PREPROCESS_MODE"] == 1:
        # | -1, +1 | --> [0, 1]
        return tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)
    elif HPS["PREPROCESS_MODE"] == 2:
        # | -2, +2 | --> [0, 1]
        return tf.clip_by_value(x / 4 + 0.5, 0.0, 1.0)
    elif HPS["PREPROCESS_MODE"] == 3:
        # | -0.5, +0.5 | --> [0, 1]
        return tf.clip_by_value(x + 0.5, 0.0, 1.0)
    else:
        return x


def save_images(
    images: tf.Tensor,
    rows: int,
    cols: int,
    title: str = "",
) -> None:
    images = depreprocess(images)

    fig, axs = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows), constrained_layout=True)
    axs = axs.flat

    for i, image in enumerate(images):
        axs[i].imshow(image.numpy())
        axs[i].set_axis_off()

    fig.suptitle(title)

    plt.savefig(f"{HPS['SAMPLE_DIR']}/{title.lower().replace(' ' , '_')}.png")
    plt.close(fig)


def save_samples(temps: List[float], epoch: str = "") -> None:
    rows = len(temps)
    cols = z_samples.shape[0]
    fig, axs = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows), constrained_layout=True)

    for i, temp in enumerate(temps):
        images = glow(z_samples * temp, training=False)
        images = depreprocess(images)

        axs[i, 0].set_title(f"Temp = {temp:.2f}")

        for j, image in enumerate(images):
            axs[i, j].imshow(image.numpy())
            axs[i, j].set_axis_off()

    fig.suptitle(f"Epoch {epoch}")

    plt.savefig(f"{HPS['SAMPLE_DIR']}/{epoch}.png")
    plt.close(fig)


@tf.function
def train_step(images: tf.Tensor) -> float:
    # if HPS["PREPROCESS_MODE"] == 3:  # FIXME
    images += tf.random.uniform(images.shape, 0.0, 1.0 / HPS["N_BINS"])

    with tf.GradientTape() as tape:
        loss = loss_object(px, images)

    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Or optimizer.minimize(loss, trainable_vars)

    return tf.reduce_mean(loss)
    # epoch_train_loss(loss)


@tf.function
def test_step(images: tf.Tensor) -> float:
    test_loss = loss_object(px, images)

    return tf.reduce_mean(test_loss)
    # epoch_test_loss(test_loss)


# ----- Start -----
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f"{now = }")

    cmd_args = arg_parsing()
    print(f"{cmd_args = }")

    # HYPER PARAMETERS
    HPS = {
        "DATASET": cmd_args.dataset,
        "TRAIN_SLICE": cmd_args.train_slice,
        "TEST_SLICE": cmd_args.test_slice,
        "IMAGE_SIZE": cmd_args.image_size,
        "CENTRAL_CROP_FRACTION": cmd_args.central_crop_fraction,
        "PREPROCESS_MODE": cmd_args.preprocess_mode,
        "N_BITS": cmd_args.n_bits,
        "N_BINS": ...,
        "DISABLE_TQDM": cmd_args.not_disable_tqdm,
        # -----
        "OUTPUT_SHAPE": ...,
        "NUM_GLOW_BLOCKS": cmd_args.num_glow_blocks,
        "NUM_STEPS_PER_BLOCK": cmd_args.num_steps_per_block,
        "COUPLING_BIJECTOR_FN": cmd_args.coupling_bijector_fn,
        "EXIT_BIJECTOR_FN": "GlowDefaultExitNetwork",
        "GRAB_AFTER_BLOCK": cmd_args.grab_after_block,
        "USE_ACTNORM": cmd_args.not_use_actnorm,
        "SEED": cmd_args.seed,
        # -----
        "EPOCHS": cmd_args.epochs,
        "BATCH_SIZE": cmd_args.batch_size,
        "LOSS": "negll_per_dim",
        "LEARNING_RATE": cmd_args.learning_rate,
        "OPTIMIZER_CLIPVALUE": cmd_args.optimizer_clipvalue,
        "OPTIMIZER_GLOBAL_CLIPNORM": cmd_args.optimizer_global_clipnorm,
        "ADAM_EPSILON": cmd_args.adam_epsilon,
        "COSINE_DECAY_STEPS": cmd_args.cosine_decay_steps,
        "COSINE_DECAY_T_MUL": cmd_args.cosine_decay_t_mul,
        "COSINE_DECAY_M_MUL": cmd_args.cosine_decay_m_mul,
        "COSINE_DECAY_ALPHA": cmd_args.cosine_decay_alpha,
        # "TEMPERATURE": 0.7,  # Not used currently  # For testing a lower temp is probably better
        # -----
        "CHECKPOINT_FREQ": cmd_args.checkpoint_freq,
        # "PRINT_FREQ": 100,
        # "TEST_PRINT_FREQ": ...,
        # -----
        "NAME": cmd_args.name,
        "ID": ...,
        # -----
        "DIR": f"{ROOT_DIR}/glow",
        "TFDS_DATA_DIR": f"{ROOT_DIR}/data",
        "SAFE_DIR": ...,
        "HYPERPARAMETERS_DIR": ...,
        "CHECKPOINT_DIR": ...,
        "EPOCH_RESULTS_DIR": ...,
        # "TRAIN_LOSSES_TXT": ...,
        # "TEST_LOSSES_TXT": ...,
        "SAMPLE_DIR": ...,
    }

    HPS["N_BINS"] = 2 ** HPS["N_BITS"]

    HPS["OUTPUT_SHAPE"] = (HPS["IMAGE_SIZE"], HPS["IMAGE_SIZE"], 3)

    # HPS["COSINE_DECAY_STEPS"] = max(HPS["EPOCHS"] - 1, 1)

    # HPS["TEST_PRINT_FREQ"] = HPS["PRINT_FREQ"] // 10

    HPS["ID"] = f"{HPS['NAME']}_{now}"

    # HPS["TFDS_DATA_DIR"] = f"{HPS['DIR']}/tfds_data_dir"
    HPS["SAFE_DIR"] = f"{HPS['DIR']}/safe/{HPS['ID']}"

    HPS["HYPERPARAMETERS_DIR"] = f"{HPS['SAFE_DIR']}/hyperparameters.json"
    HPS["CHECKPOINT_DIR"] = f"{HPS['SAFE_DIR']}/checkpoints/ckpt"
    HPS["EPOCH_RESULTS_DIR"] = f"{HPS['SAFE_DIR']}/epoch_results.csv"
    # HPS["TRAIN_LOSSES_TXT"] = f"{HPS['SAFE_DIR']}/train_losses.txt"
    # HPS["TEST_LOSSES_TXT"] = f"{HPS['SAFE_DIR']}/test_losses.txt"
    HPS["SAMPLE_DIR"] = f"{HPS['SAFE_DIR']}/samples"

    os.makedirs(HPS["SAFE_DIR"], exist_ok=True)
    os.makedirs(HPS["SAMPLE_DIR"], exist_ok=True)

    with open(HPS["HYPERPARAMETERS_DIR"], "w") as json_file:
        json.dump(HPS, json_file, indent=4)

    tf.random.set_seed(HPS["SEED"])

    plt.rcParams["figure.dpi"] = 100
    plt.style.use("seaborn-darkgrid")

    (train_data, test_data), info = tfds.load(
        HPS["DATASET"],
        split=[HPS["TRAIN_SLICE"], HPS["TEST_SLICE"]],
        with_info=True,
        data_dir=HPS["TFDS_DATA_DIR"],
    )
    # Images are of type tf.uint8

    train_data = train_data.shuffle(len(train_data), reshuffle_each_iteration=True)

    train_data = train_data.map(preprocess).batch(HPS["BATCH_SIZE"])
    test_data = test_data.map(preprocess).batch(HPS["BATCH_SIZE"])

    x = next(iter(train_data))

    coupling = None
    if cmd_args.coupling_bijector_fn == "GlowDefaultNetwork":
        coupling = tfb.GlowDefaultNetwork
    elif cmd_args.coupling_bijector_fn == "GlowAdditiveNetwork":
        coupling = GlowAdditiveNetwork

    glow = tfb.Glow(
        output_shape=HPS["OUTPUT_SHAPE"],
        num_glow_blocks=HPS["NUM_GLOW_BLOCKS"],
        num_steps_per_block=HPS["NUM_STEPS_PER_BLOCK"],
        coupling_bijector_fn=coupling,
        exit_bijector_fn=tfb.GlowDefaultExitNetwork,
        grab_after_block=HPS["GRAB_AFTER_BLOCK"],
        use_actnorm=HPS["USE_ACTNORM"],
        seed=HPS["SEED"],
        validate_args=True,
    )

    z_shape = glow.inverse_event_shape(HPS["OUTPUT_SHAPE"])

    pz = tfd.Sample(tfd.Normal(0.0, 1.0), z_shape)

    # Calling glow on distribution p(z) creates our glow distribution over images.
    px = glow(pz)

    checkpoint = tf.train.Checkpoint(glow)

    # Add noise as done during training
    x += tf.random.uniform(x.shape, 0.0, 1.0 / HPS["N_BINS"])
    if HPS["BATCH_SIZE"] >= 8:
        save_images(x, int(HPS["BATCH_SIZE"] / 8), 8, title="Dataset")
    else:
        save_images(x, 1, HPS["BATCH_SIZE"], title="Dataset")

    images = px.sample(4)
    save_images(images, 1, 4, title="Initial Samples")

    # Samples fixed z_values for image generation during training
    z_samples = pz.sample(10)

    print(f"{images.shape = }")
    print(f"{z_samples.shape = }")
    print(f"{px = }")
    print(f"{pz = }")

    loss_object = None
    if HPS["LOSS"] == "negll":
        negll_loss = lambda dist, data: -dist.log_prob(data)
        loss_object = negll_loss
    elif HPS["LOSS"] == "negll_per_dim":
        num_dimensions = np.prod(HPS["OUTPUT_SHAPE"])  # Should be equal to z_shape
        print(f"{num_dimensions = }")
        negll_per_dim_loss = (
            lambda dist, data: -tf.reduce_mean(dist.log_prob(data), axis=-1) / num_dimensions
        )
        loss_object = negll_per_dim_loss
        # Probably calculates nats per dimension
        # You would neet to divide by log(2) * num_dimensions to get bits

    lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        HPS["LEARNING_RATE"],
        HPS["COSINE_DECAY_STEPS"],
        t_mul=HPS["COSINE_DECAY_T_MUL"],
        m_mul=HPS["COSINE_DECAY_M_MUL"],
        alpha=HPS["COSINE_DECAY_ALPHA"],
    )
    optimizer = tf.keras.optimizers.Adam(
        lr_scheduler(0),
        epsilon=HPS["ADAM_EPSILON"],
        clipvalue=HPS["OPTIMIZER_CLIPVALUE"],
        global_clipnorm=HPS["OPTIMIZER_GLOBAL_CLIPNORM"],
    )

    train_losses = []
    test_losses = []

    df = pd.DataFrame(
        columns=[
            "epoch",
            "train_mean",
            "train_std",
            "train_min",
            "train_q25",
            "train_q75",
            "train_max",
            "test_mean",
            "test_std",
            "test_min",
            "test_q25",
            "test_q75",
            "test_max",
            "learning_rate",
            "timestamp",
        ]
    )

    # Clears all TXT files
    # open(HPS["TRAIN_LOSSES_TXT"], "w").close()
    # open(HPS["TEST_LOSSES_TXT"], "w").close()

    trainable_vars = glow.trainable_variables

    for epoch in trange(HPS["EPOCHS"], desc="Epoch", position=0, leave=True):
        # Reset the metrics at the start of the next epoch
        # epoch_train_loss.reset_states()
        # epoch_test_loss.reset_states()

        optimizer.learning_rate = lr_scheduler(epoch)

        for i, images in enumerate(
            tqdm(train_data, desc="Training", position=1, leave=False, disable=HPS["DISABLE_TQDM"])
        ):
            l = train_step(images)
            train_losses.append(l)

            # if i % HPS["PRINT_FREQ"] == 0:
            #    with open(HPS["TRAIN_LOSSES_TXT"], "a") as file:
            #        file.write(f"{l}\n")

        # If loss is nan the model won't recover from it and we can stop the training
        if tf.math.is_nan(l):
            print("Loss was nan --> Aborting")
            break

        for j, test_images in enumerate(
            tqdm(test_data, desc="Testing", position=2, leave=False, disable=HPS["DISABLE_TQDM"])
        ):
            l = test_step(test_images)
            test_losses.append(l)

            # if j % HPS["TEST_PRINT_FREQ"] == 0:
            #    with open(HPS["TEST_LOSSES_TXT"], "a") as file:
            #        file.write(f"{l}\n")

        if epoch % HPS["CHECKPOINT_FREQ"] == 0:
            checkpoint.save(HPS["CHECKPOINT_DIR"])
            df.to_csv(HPS["EPOCH_RESULTS_DIR"], na_rep="nan")
            save_samples([0.25, 0.5, 0.75, 1.0], epoch=str(epoch))

        epoch_results = [
            epoch,
            np.mean(train_losses),
            np.std(train_losses),
            np.amin(train_losses),
            np.quantile(train_losses, 0.25),
            np.quantile(train_losses, 0.75),
            np.amax(train_losses),
            np.mean(test_losses),
            np.std(test_losses),
            np.amin(test_losses),
            np.quantile(test_losses, 0.25),
            np.quantile(test_losses, 0.75),
            np.amax(test_losses),
            optimizer.learning_rate.numpy(),
            datetime.datetime.now().timestamp(),
        ]
        print(f"{epoch_results = }")

        df.loc[len(df)] = epoch_results

        train_losses = []
        test_losses = []

        # epoch_train_losses.append(epoch_train_loss.result())
        # epoch_test_losses.append(epoch_test_loss.result())

    # After training
    checkpoint.save(HPS["CHECKPOINT_DIR"])
    df.to_csv(HPS["EPOCH_RESULTS_DIR"], na_rep="nan")
    save_samples([0.25, 0.5, 0.75, 1.0], epoch=str(epoch))
