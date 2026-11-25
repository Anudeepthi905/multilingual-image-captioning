"""Show, Attend and Tell: CNN encoder, Bahdanau attention and a GRU decoder."""
import tensorflow as tf

from src import config


class CNN_Encoder(tf.keras.Model):
    """Projects cached CNN features (batch, locations, 2048) to (batch, locations, 256)."""

    def __init__(self, embedding_dim: int = config.EMBEDDING_DIM):
        super().__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation="relu")

    def call(self, x):
        return self.fc(x)


class BahdanauAttention(tf.keras.layers.Layer):
    """Additive attention: score = V(tanh(W1(features) + W2(hidden)))."""

    def __init__(self, units: int):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch, locations, emb); hidden: (batch, units)
        hidden_with_time = tf.expand_dims(hidden, 1)                       # (b, 1, units)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time)))  # (b, L, 1)
        attention_weights = tf.nn.softmax(score, axis=1)                    # over locations
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)  # (b, emb)
        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    """Embedding -> [context ; embedding] -> GRU -> Dense -> Dense(vocab)."""

    def __init__(self, embedding_dim: int, units: int, vocab_size: int):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                       recurrent_initializer="glorot_uniform")
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)                                               # (b, 1, emb)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)      # (b, 1, 2*emb)
        output, state = self.gru(x)
        x = self.fc1(output)                                                # (b, 1, units)
        x = tf.reshape(x, (-1, x.shape[2]))
        return self.fc2(x), state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


_xent = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def loss_function(real, pred):
    """Cross-entropy that ignores padding (token id 0)."""
    mask = tf.cast(tf.math.not_equal(real, 0), pred.dtype)
    loss = _xent(real, pred) * mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


def build_models(vocab_size: int, embedding_dim: int = config.EMBEDDING_DIM,
                 units: int = config.UNITS, locations: int = 64):
    """Create encoder/decoder and run one dummy step so all variables exist."""
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    feats = encoder(tf.zeros((1, locations, 2048)))
    decoder(tf.zeros((1, 1), tf.int64), feats, decoder.reset_state(1))
    return encoder, decoder
