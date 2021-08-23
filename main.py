import os

import tensorflow as tf
import tensorflow_hub as hub
from wav2vec2 import Wav2Vec2Config

config = Wav2Vec2Config()

print("TF version:", tf.__version__)


pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)

AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2


inputs = tf.keras.Input(shape=(AUDIO_MAXLEN,))
hidden_states = pretrained_layer(inputs)
outputs = tf.keras.layers.Dense(config.vocab_size)(hidden_states)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model(tf.random.uniform(shape=(BATCH_SIZE, AUDIO_MAXLEN)))
model.summary()

