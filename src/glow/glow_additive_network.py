import functools

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


class GlowAdditiveNetwork(tfk.Sequential):
    """
    Adapted from GlowDefaultNetwork (https://github.com/tensorflow/probability/blob/v0.18.0/tensorflow_probability/python/bijectors/glow.py#L842-L873).
    Additive network for the glow bijector.
    This builds a 3 layer convolutional network, with relu activation functions
    and he_normal initializer. The first and third layers have default kernel
    shape of 3, and the second layer is a 1x1 convolution. This is the setup
    in the public version of Glow.
    The output of the convolutional network defines the components of an Additive
    transformation (i.e. y = x + b), where x and b are all tensors of
    the same shape.
    """

    def __init__(self, input_shape, num_hidden=400, kernel_shape=3):
        """Additive network for glow bijector."""
        # Only shift, so c outputs.
        this_nchan = input_shape[-1]
        conv_last = functools.partial(
            tfkl.Conv2D,
            padding="same",
            kernel_initializer=tf.initializers.zeros(),
            bias_initializer=tf.initializers.zeros(),
        )
        super(GlowAdditiveNetwork, self).__init__(
            [
                tfkl.Input(shape=input_shape),
                tfkl.Conv2D(
                    num_hidden,
                    kernel_shape,
                    padding="same",
                    kernel_initializer=tf.initializers.he_normal(),
                    activation="relu",
                ),
                tfkl.Conv2D(
                    num_hidden,
                    1,
                    padding="same",
                    kernel_initializer=tf.initializers.he_normal(),
                    activation="relu",
                ),
                conv_last(this_nchan, kernel_shape),
            ]
        )


if __name__ == "__main__":
    glow_additive_network = GlowAdditiveNetwork((32, 32, 3))
    print(glow_additive_network)
    