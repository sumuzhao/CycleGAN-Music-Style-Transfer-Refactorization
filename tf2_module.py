import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Input
from collections import namedtuple


def abs_criterion(pred, target):
    return tf.reduce_mean(tf.abs(pred - target))


def mae_criterion(pred, target):
    return tf.reduce_mean((pred - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def softmax_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def padding(x, p=3):
    return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")


def instance_norm(x, epsilon=1e-5):

    scale = tf.Variable(initial_value=np.random.normal(1., 0.02, x.shape[-1:]),
                        trainable=True,
                        name='SCALE',
                        dtype=tf.float32)
    offset = tf.Variable(initial_value=np.zeros(x.shape[-1:]),
                         trainable=True,
                         name='OFFSET',
                         dtype=tf.float32)
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (x - mean) * inv
    return scale * normalized + offset


def resnet_block(x, dim, k_init, ks=3, s=1):

    # e.g, x is (batch * 128 * 128 * 3)
    p = (ks - 1) // 2
    # For ks = 3, p = 1
    y = layers.Lambda(padding,
                      arguments={'p': p},
                      name='PADDING_1')(x)
    # After first padding, (batch * 130 * 130 * 3)

    y = layers.Conv2D(filters=dim,
                      kernel_size=ks,
                      strides=s,
                      padding='valid',
                      kernel_initializer=k_init,
                      use_bias=False)(y)
    y = layers.Lambda(instance_norm,
                      name='IN')(y)
    y = layers.ReLU()(y)
    # After first conv2d, (batch * 128 * 128 * 3)

    y = layers.Lambda(padding,
                      arguments={'p': p},
                      name='PADDING_2')(y)
    # After second padding, (batch * 130 * 130 * 3)

    y = layers.Conv2D(filters=dim,
                      kernel_size=ks,
                      strides=s,
                      padding='valid',
                      kernel_initializer=k_init,
                      use_bias=False)(y)
    y = layers.Lambda(instance_norm,
                      name='IN')(y)
    y = layers.ReLU()(y + x)
    # After second conv2d, (batch * 128 * 128 * 3)

    return y


def build_discriminator(options, name='Discriminator'):

    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = Input(shape=(options.time_step,
                          options.pitch_range,
                          options.output_nc))

    x = inputs

    x = layers.Conv2D(filters=options.df_dim,
                      kernel_size=7,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 32 * 42 * 64)

    x = layers.Conv2D(filters=options.df_dim * 4,
                      kernel_size=7,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_2')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 16 * 21 * 256)

    x = layers.Conv2D(filters=1,
                      kernel_size=7,
                      strides=1,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_3')(x)
    # (batch * 16 * 21 * 1)

    outputs = x

    return Model(inputs=inputs,
                 outputs=outputs,
                 name=name)


def build_generator(options, name='Generator'):

    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = Input(shape=(options.time_step,
                          options.pitch_range,
                          options.output_nc))

    x = inputs
    # (batch * 64 * 84 * 1)

    x = layers.Lambda(padding,
                      name='PADDING_1')(x)
    # (batch * 70 * 90 * 1)

    x = layers.Conv2D(filters=options.gf_dim,
                      kernel_size=7,
                      strides=1,
                      padding='valid',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_1')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_1')(x)
    x = layers.ReLU()(x)
    # (batch * 64 * 84 * 64)

    x = layers.Conv2D(filters=options.gf_dim * 2,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_2')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_2')(x)
    x = layers.ReLU()(x)
    # (batch * 32 * 42 * 128)

    x = layers.Conv2D(filters=options.gf_dim * 4,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_3')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_3')(x)
    x = layers.ReLU()(x)
    # (batch * 16 * 21 * 256)

    for i in range(10):
        # x = resnet_block(x, options.gf_dim * 4)
        x = layers.Lambda(resnet_block,
                          arguments={'dim': options.gf_dim * 4,
                                     'k_init': initializer},
                          name='ResNet_Block_{}'.format(i))(x)
    # (batch * 16 * 21 * 256)

    x = layers.Conv2DTranspose(filters=options.gf_dim * 2,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False,
                               name='DECONV2D_1')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_4')(x)
    x = layers.ReLU()(x)
    # (batch * 32 * 42 * 128)

    x = layers.Conv2DTranspose(filters=options.gf_dim,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False,
                               name='DECONV2D_2')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_5')(x)
    x = layers.ReLU()(x)
    # (batch * 64 * 84 * 64)

    x = layers.Lambda(padding,
                      name='PADDING_2')(x)
    # After padding, (batch * 70 * 90 * 64)

    x = layers.Conv2D(filters=options.output_nc,
                      kernel_size=7,
                      strides=1,
                      padding='valid',
                      kernel_initializer=initializer,
                      activation='sigmoid',
                      use_bias=False,
                      name='CONV2D_4')(x)
    # (batch * 64 * 84 * 1)

    outputs = x

    return Model(inputs=inputs,
                 outputs=outputs,
                 name=name)


def build_discriminator_classifier(options, name='Discriminator_Classifier'):

    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = Input(shape=(options.time_step,
                          options.pitch_range,
                          options.output_nc))

    x = inputs
    # (batch * 64, 84, 1)

    x = layers.Conv2D(filters=options.df_dim,
                      kernel_size=[1, 12],
                      strides=[1, 12],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 64 * 7 * 64)

    x = layers.Conv2D(filters=options.df_dim * 2,
                      kernel_size=[4, 1],
                      strides=[4, 1],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_2')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 16 * 7 * 128)

    x = layers.Conv2D(filters=options.df_dim * 4,
                      kernel_size=[2, 1],
                      strides=[2, 1],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_3')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_2')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 8 * 7 * 256)

    x = layers.Conv2D(filters=options.df_dim * 8,
                      kernel_size=[8, 1],
                      strides=[8, 1],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_4')(x)
    x = layers.Lambda(instance_norm,
                      name='IN_3')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 1 * 7 * 512)

    x = layers.Conv2D(filters=2,
                      kernel_size=[1, 7],
                      strides=[1, 7],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_5')(x)
    # (batch * 1 * 1 * 2)

    x = tf.reshape(x, [-1, 2])
    # (batch * 2)

    outputs = x

    return Model(inputs=inputs,
                 outputs=outputs,
                 name=name)


if __name__ == '__main__':

    OPTIONS = namedtuple('OPTIONS', 'batch_size '
                                    'time_step '
                                    'input_nc '
                                    'output_nc '
                                    'pitch_range '
                                    'gf_dim '
                                    'df_dim ')
    options = OPTIONS._make((128,
                             64,
                             1,
                             1,
                             84,
                             64,
                             64))

    model = build_generator(options)
    print(model.summary())
