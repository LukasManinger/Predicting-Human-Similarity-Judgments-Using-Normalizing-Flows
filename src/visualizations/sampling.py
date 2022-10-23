import json
import os

import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow import keras
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

sns.set_theme("paper", "whitegrid", "Dark2", "DejaVu Sans", rc={"figure.dpi": 100})

PATH = "/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/safe/128-2_2022-09-20_17:36:27/hyperparameters.json"
NUMBER = 11

def main():
    #tf.random.set_seed(2410)

    glow, pz, px = load_model(PATH, NUMBER)

    name = PATH.replace("/work/scratch/lm83qyjo/bt2/bachelor-thesis/bachelor_thesis/tfp-glow/safe/", "", 1).replace("/hyperparameters.json", "", 1)
    
    #for i in range(3):
    save_samples_3(glow, pz, px)

def load_model(path, number):
    hps = dict()
    with open(path, "r") as json_file:
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

    status = checkpoint.restore(f"{hps['CHECKPOINT_DIR']}-{number}")
    status.assert_consumed().assert_existing_objects_matched().assert_nontrivial_match()

    return glow, pz, px

def save_samples(glow, pz, px, name: str = "") -> None:
    depreprocess = lambda t :tf.clip_by_value(t + 0.5, 0.0, 1.0)

    temps = [0.25, 0.5, 0.75, 1.0]
    samples_per_temp = 8
    
    rows = len(temps)
    cols = samples_per_temp
    fig, axs = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows), tight_layout=True)

    for i, temp in enumerate(temps):
        images = glow(pz.sample(samples_per_temp) * temp, training=False)
        # pt = tfb.Power(temp**2)(px)
        # images = pt.sample(samples_per_temp)
        images = depreprocess(images)

        axs[i, 0].set_title(f"Temp = {temp:.2f}")

        for j, image in enumerate(images):
            axs[i, j].imshow(image.numpy())
            # axs[i, j].set_axis_off()
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    fig.suptitle(name)

    plt.savefig(f"analysis/out/samples/{name}.png")
    plt.close(fig)

def save_samples_2(glow, pz, px, name: str = "") -> None:
    depreprocess = lambda t :tf.clip_by_value(t + 0.5, 0.0, 1.0)

    temp = 0.75
    #samples_per_temp = 8
    n = 6
    m = 3
    #rows = len(temps)
    #cols = samples_per_temp
    fig, axs = plt.subplots(m, n, figsize=(7.0866, 3.0), tight_layout=True)

    for i, row in enumerate(range(m)):
        images = glow(pz.sample(n) * temp, training=False)
        images = depreprocess(images)

        for j, image in enumerate(images):
            axs[i, j].imshow(image.numpy())
            # axs[i, j].set_axis_off()
            axs[i, j].set_xticks([])
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticks([])
            axs[i, j].set_yticklabels([])
    
    fig.subplots_adjust(wspace=0, hspace=0)
    
    plt.savefig(f"analysis/out/samples/{name}.png")
    plt.savefig(f"analysis/out/samples/{name}.pdf")
    #plt.close(fig)

def save_samples_3(glow, pz, px) -> None:
    depreprocess = lambda t :tf.clip_by_value(t + 0.5, 0.0, 1.0)

    temp = 0.75
    n = 8 * 4

    images = glow(pz.sample(n) * temp, training=False)
    images = depreprocess(images)

    for i, image in enumerate(images):
        keras.utils.save_img(f"analysis/out/samples/spam/{i}.png", image.numpy() * 255, scale=False)


if __name__ == "__main__":
    main()
