import numpy as np
import pytest
import tensorflow as tf

from src.model import BahdanauAttention, CNN_Encoder, RNN_Decoder, loss_function


@pytest.mark.parametrize("locations", [64, 49])
def test_attention_output_shapes(locations):
    batch, emb, units = 2, 256, 512
    attn = BahdanauAttention(units)
    features = tf.random.normal((batch, locations, emb))
    hidden = tf.random.normal((batch, units))
    context, weights = attn(features, hidden)
    assert context.shape == (batch, emb)
    assert weights.shape == (batch, locations, 1)
    np.testing.assert_allclose(tf.reduce_sum(weights, axis=1).numpy(), 1.0, rtol=1e-5)


def test_encoder_and_decoder_shapes():
    vocab, batch = 100, 3
    enc = CNN_Encoder(256)
    dec = RNN_Decoder(256, 512, vocab)
    feats = enc(tf.random.normal((batch, 64, 2048)))
    assert feats.shape == (batch, 64, 256)
    logits, state, weights = dec(tf.ones((batch, 1), tf.int64), feats, dec.reset_state(batch))
    assert logits.shape == (batch, vocab)
    assert state.shape == (batch, 512)
    assert weights.shape == (batch, 64, 1)


def test_decoder_carries_hidden_state_between_steps():
    """The GRU must start from the hidden state it is given, not from zeros."""
    tf.keras.utils.set_random_seed(0)
    enc = CNN_Encoder(256)
    dec = RNN_Decoder(256, 512, 50)
    feats = enc(tf.random.normal((1, 64, 2048)))
    word = tf.ones((1, 1), tf.int64)
    dec(word, feats, dec.reset_state(1))  # build the layers
    # Cut the attention path (W2 sees the hidden state), so the hidden state can
    # only reach the output through the GRU's own memory.
    for w in dec.attention.W2.weights:
        w.assign(tf.zeros_like(w))
    _, s_zero, _ = dec(word, feats, tf.zeros((1, 512)))
    _, s_rand, _ = dec(word, feats, tf.random.normal((1, 512)))
    assert not np.allclose(s_zero.numpy(), s_rand.numpy())


def test_loss_ignores_padding():
    logits = tf.constant([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    real_with_pad = tf.constant([0, 1])  # first position is padding
    only_real = loss_function(tf.constant([1]), logits[1:])
    assert float(loss_function(real_with_pad, logits)) == pytest.approx(float(only_real))
