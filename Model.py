import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras import layers, models
import HelpFunc as hf


def SRCNN():
    f1 = 9
    n1 = 64
    f2 = 1
    n2 = 32
    f3 = 5
    c = 3

    X = tf.keras.Input(shape=(None, None, c))

    patch_extraction = layers.Conv2D(
        filters=n1,
        kernel_size=(f1, f1),
        padding='valid',
        activation='relu',
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
        bias_initializer=Zeros(),
        name="patch_extraction"
    )

    nonlinear_map = layers.Conv2D(
        filters=n2,
        kernel_size=(f2, f2),
        padding='valid',
        activation='relu',
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
        bias_initializer=Zeros(),
        name="nonlinear_map"
    )

    reconstruction = layers.Conv2D(
        filters=c,
        kernel_size=(f3, f3),
        padding='valid',
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
        bias_initializer=Zeros(),
        name="reconstuction"
    )

    patch_extrac_layer = patch_extraction(X)
    nonlinear_map_layer = nonlinear_map(patch_extrac_layer)
    recon_tmp_layer = reconstruction(nonlinear_map_layer)
    recon_layer = tf.clip_by_value(recon_tmp_layer, 0.0, 1.0)

    model = models.Model(inputs=X, outputs=recon_layer, name="srcnn")

    # model.compile(
    #     optimizer=tf.keras.optimizers.SGD(
    #         learning_rate=1e-5, momentum=0.9, name='SGD'),
    #     loss=tf.keras.losses.MeanSquaredError(),
    #     metrics=[hf.PSNR]
    # )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[hf.PSNR]
    )

    return model
