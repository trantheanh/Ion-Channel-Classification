import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from optimizers.core import build_optimizer
from metrics.core import BinarySpecificity, BinarySensitivity, BinaryMCC, BinaryAccuracy, BinaryF1Score


def flood_loss(b=0.05):
    def get_loss(y_true, y_pred):
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        loss = keras.backend.abs(loss - b) + b
        return loss
    return get_loss


def get_metrics(threshold=0.5):

    return [
        BinaryF1Score(threshold),
        BinaryAccuracy(threshold),
        BinaryMCC(threshold),
        BinarySensitivity(threshold),
        BinarySpecificity(threshold)
    ]


"""# Build model"""


def build_lstm_maxout(hparams):
    keras.backend.clear_session()
    rnn_input = layers.Input(shape=(15, 20))
    mlp_input = layers.Input(shape=(30,))

    rnn_imd = rnn_input
    for i in range(hparams["rnn_layers"]):
      rnn_imd = layers.LSTM(
          units=hparams["rnn_units"],
          return_sequences=(i+1 < hparams["rnn_layers"]),
          # activation="sigmoid"
      )(rnn_imd)

    mlp_imd = []
    for i in range(hparams["maxout_head"]):
        mlp_imd.append(layers.Dense(units=hparams["maxout_units"])(mlp_input))

    if hparams["maxout_head"] > 1:
        mlp_imd = layers.Maximum()(mlp_imd)
    elif hparams["maxout_head"] == 1:
        mlp_imd = layers.Activation(activation="relu")(mlp_imd[0])
    else:
        mlp_imd = mlp_input

    imd = layers.Concatenate(axis=-1)([rnn_imd, mlp_imd])

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[mlp_input, rnn_input], outputs=output_tf)

    model.compile(
      optimizer=build_optimizer(
          optimizer_name=hparams["optimizer"],
          learning_rate=hparams["learning_rate"],
          decay=hparams["lr_decay"]
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=get_metrics(hparams["threshold"])
    )

    return model


def build_lstm(hparams):
    keras.backend.clear_session()
    rnn_input = layers.Input(shape=(15, 20))
    mlp_input = layers.Input(shape=(30,))

    rnn_imd = rnn_input
    for i in range(hparams["rnn_layers"]):
        rnn_imd = layers.LSTM(
            units=hparams["rnn_units"],
            return_sequences=(i+1 < hparams["rnn_layers"]),
            # activation="sigmoid"
        )(rnn_imd)

    imd = rnn_imd

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[mlp_input, rnn_input], outputs=output_tf)

    model.compile(
      optimizer=build_optimizer(
          optimizer_name=hparams["optimizer"],
          learning_rate=hparams["learning_rate"],
          decay=hparams["lr_decay"]
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=get_metrics(hparams["threshold"])
    )

    return model


def build_lstm_conv(hparams):
    keras.backend.clear_session()
    rnn_input = layers.Input(shape=(15, 20))
    mlp_input = layers.Input(shape=(30,))

    rnn_imd = rnn_input
    rnn_imd = layers.Conv1D(
        filters=hparams["conv1d_depth"],
        kernel_size=hparams["conv1d_size"],
        strides=hparams["conv1d_stride"],
        padding='SAME',
        activation="relu"
    )(rnn_imd)
    for i in range(hparams["rnn_layers"]):
        rnn_imd = layers.LSTM(
            units=hparams["rnn_units"],
            return_sequences=(i+1 < hparams["rnn_layers"]),
            # activation="sigmoid"
        )(rnn_imd)

    imd = rnn_imd
    imd = layers.Dropout(rate=hparams["dropout"])(imd)

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[mlp_input, rnn_input], outputs=output_tf)

    model.compile(
      optimizer=build_optimizer(
          optimizer_name=hparams["optimizer"],
          learning_rate=hparams["learning_rate"],
          decay=hparams["lr_decay"]
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=get_metrics(hparams["threshold"])
    )

    return model


def build_lstm_maxout_dropout(hparams):
    keras.backend.clear_session()
    rnn_input = layers.Input(shape=(15, 20))
    mlp_input = layers.Input(shape=(30,))

    rnn_imd = rnn_input
    for i in range(hparams["rnn_layers"]):
      rnn_imd = layers.LSTM(
          units=hparams["rnn_units"],
          return_sequences=(i+1 < hparams["rnn_layers"]),
          # activation="sigmoid"
      )(rnn_imd)

    # mlp_imd = []
    # for i in range(hparams["maxout_head"]):
    #     mlp_imd.append(layers.Dense(units=hparams["maxout_units"])(mlp_input))

    # if hparams["maxout_head"] > 1:
    #     mlp_imd = layers.Maximum()(mlp_imd)
    # elif hparams["maxout_head"] == 1:
    #     mlp_imd = layers.Activation(activation="relu")(mlp_imd[0])
    # else:
    #     mlp_imd = mlp_input

    mlp_imd = layers.Dense(units=hparams["maxout_units"], activation="tanh")(mlp_input)

    imd = layers.Concatenate(axis=-1)([rnn_imd, mlp_imd])
    imd = layers.Dropout(rate=hparams["dropout"])(imd)

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[mlp_input, rnn_input], outputs=output_tf)

    model.compile(
      optimizer=build_optimizer(
          optimizer_name=hparams["optimizer"],
          learning_rate=hparams["learning_rate"],
          decay=hparams["lr_decay"]
      ),
      loss=flood_loss(hparams["flood_loss_coef"]),#keras.losses.binary_crossentropy,
      metrics=get_metrics(hparams["threshold"])
    )

    return model


def build_conv_lstm_maxout_dropout(hparams):
    keras.backend.clear_session()
    rnn_input = layers.Input(shape=(15, 20))
    mlp_input = layers.Input(shape=(30,))

    rnn_imd = rnn_input
    rnn_imd = layers.Conv1D(
        filters=hparams["conv1d_depth"],
        kernel_size=hparams["conv1d_size"],
        strides=hparams["conv1d_stride"],
        padding='SAME',
        activation="relu"
    )(rnn_imd)
    for i in range(hparams["rnn_layers"]):
        rnn_imd = layers.LSTM(
            units=hparams["rnn_units"],
            return_sequences=(i+1 < hparams["rnn_layers"]),
        )(rnn_imd)

    mlp_imd = []
    for i in range(hparams["maxout_head"]):
        mlp_imd.append(layers.Dense(units=hparams["maxout_units"])(mlp_input))

    if hparams["maxout_head"] > 1:
        mlp_imd = layers.Maximum()(mlp_imd)
    elif hparams["maxout_head"] == 1:
        mlp_imd = layers.Activation(activation="relu")(mlp_imd[0])
    else:
        mlp_imd = mlp_input

    imd = layers.Concatenate(axis=-1)([rnn_imd, mlp_imd])
    imd = layers.Dropout(rate=hparams["dropout"])(imd)

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[mlp_input, rnn_input], outputs=output_tf)

    model.compile(
      optimizer=build_optimizer(
          optimizer_name=hparams["optimizer"],
          learning_rate=hparams["learning_rate"],
          decay=hparams["lr_decay"]
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=get_metrics(hparams["threshold"])
    )

    return model
