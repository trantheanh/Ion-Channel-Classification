import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from optimizers.core import build_optimizer
from metrics.core import BinarySpecificity, BinarySensitivity, BinaryMCC, BinaryAccuracy, BinaryF1Score
from constant.shape import InputShape
from sklearn.feature_extraction.text import TfidfVectorizer
from resource import RESOURCE_PATH
import pickle
import os
from constant.url import DataPath


def flood_loss(b=0.05):
    def get_loss(y_true, y_pred):
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        loss = keras.backend.abs(loss - b) + b
        return loss
    return get_loss


def get_metrics(threshold=0.5) -> list:

    return [
        BinaryF1Score(threshold),
        BinaryAccuracy(threshold),
        BinaryMCC(threshold),
        BinarySensitivity(threshold),
        BinarySpecificity(threshold)
    ]


"""# Build model"""


# PSSM_GRU = 512
# PSSM_Dropout = 0.1
# EMB_Conv1D = 32
# EMB_LSTM = 1024
# EMB_Dropout = 0.1
# TFIDF_Dropout = 0.1
# TFIDF_units = 128
# dropout = 0.3
# units = 512
# lr = 0.00001
# threshold = 0.1
# optimizer = nadam
# decay = 0
def build_pfam_emb(hparams) -> keras.models.Model:
    keras.backend.clear_session()
    emb_input = layers.Input(shape=(InputShape.EMB_LENGTH, InputShape.EMB_DIM))
    pssm_input = layers.Input(shape=(InputShape.PSSM_LENGTH, InputShape.PSSM_DIM))
    tfidf_input = layers.Input(shape=(InputShape.TFIDF_DIM,))

    # PSSM
    pssm_imd = pssm_input

    pssm_imd = layers.GRU(
        units=hparams["PSSM_GRU"],
        return_sequences=False,
        dropout=hparams["PSSM_Dropout"]
    )(pssm_imd)

    # Fasttext Emb
    emb_imd = emb_input

    emb_imd = layers.Conv1D(
        filters=hparams["EMB_Conv1D"],
        kernel_size=3,
        strides=1,
        padding='SAME',
        activation="relu"
    )(emb_imd)
    emb_imd = layers.MaxPool1D()(emb_imd)

    emb_imd = layers.LSTM(
        units=hparams["EMB_LSTM"],
        return_sequences=False,
        dropout=hparams["EMB_Dropout"]
    )(emb_imd)

    # TF-IDF
    tfidf_imd = tfidf_input
    tfidf_imd = layers.Dropout(rate=hparams["TFIDF_Dropout"])(tfidf_imd)
    tfidf_imd = layers.Dense(units=hparams["TFIDF_units"], activation="relu")(tfidf_imd)

    # Concate
    imd = layers.Concatenate(axis=-1)([emb_imd, pssm_imd, tfidf_imd])
    imd = layers.Dropout(rate=hparams["dropout"])(imd)
    imd = layers.Dense(units=hparams["units"], activation="relu")(imd)
    imd = layers.Dropout(rate=hparams["dropout"])(imd)

    output_tf = layers.Dense(
        units=1,
        activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[emb_input, pssm_input, tfidf_input], outputs=output_tf)

    model.compile(
        optimizer=build_optimizer(hparams["optimizer"], hparams["lr"], hparams["decay"]),
        loss=keras.losses.binary_crossentropy,
        metrics=get_metrics(threshold=hparams["threshold"])
    )

    return model


def train_tfidf(train_raw_data_x):
    train_raw_data_x = [" ".join(tokens[:]) for _, tokens in enumerate(train_raw_data_x)]
    vectorizer = TfidfVectorizer(analyzer="char")
    vectorizer.fit(train_raw_data_x)

    # Save tfidf_vectorizer
    pickle.dump(vectorizer, open(os.path.join(RESOURCE_PATH, DataPath.tfidf_file_name), "wb"))

    return vectorizer


