import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from optimizers.core import build_optimizer
from metrics.core import BinarySpecificity, BinarySensitivity, BinaryMCC, BinaryAccuracy


"""# Build model"""


def build_model(hparams):
    keras.backend.clear_session()
    rnn_input = layers.Input(shape=(15, 20))
    mlp_input = layers.Input(shape=(30,))

    rnn_imd = rnn_input
    for i in range(hparams["rnn_layers"]):
        rnn_imd = layers.LSTM(
          units=hparams["rnn_units"],
          return_sequences=(i+1 < hparams["rnn_layers"]),
          activation="sigmoid"
        )(rnn_imd)

    mlp_imd = []
    for i in range(hparams["maxout_head"]):
        mlp_imd.append(layers.Dense(units=hparams["maxout_units"])(mlp_input))

    # mlp_imd = layers.BatchNormalization()(mlp_imd)

    if hparams["maxout_head"] > 1:
        mlp_imd = layers.Maximum()(mlp_imd)
    elif hparams["maxout_head"] == 1:
        mlp_imd = layers.Activation(activation="relu")(mlp_imd[0])
    else:
        mlp_imd = mlp_input

    # mlp_imd = layers.Activation(activation="sigmoid")(mlp_imd)
    imd = layers.Concatenate(axis=-1)([rnn_imd, mlp_imd])
    imd = layers.Dropout(rate=hparams["dropout"])(imd)

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[mlp_input, rnn_input], outputs=output_tf)

    def compute_loss(y_true, y_pred, b=0.05):
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        loss = keras.backend.abs(loss - b) + b
        return loss

    model.compile(
      optimizer=build_optimizer(
          optimizer_name=hparams["optimizer"],
          learning_rate=hparams["learning_rate"],
          decay=hparams["lr_decay"]
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=[
        BinaryAccuracy(hparams["threshold"]),
        BinaryMCC(hparams["threshold"]),
        BinarySensitivity(hparams["threshold"]),
        BinarySpecificity(hparams["threshold"])
      ]
    )

    return model