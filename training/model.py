# -*- coding: utf-8 -*-
"""
#Model Architecture
---
This file contains the transformer architecture and couple of helper functions. 
"""

#Importing required libraries
import json
from pathlib import Path
import numpy as np
import math
from glob import glob
import tensorflow as tf
import os
import time

"""##Absolute Positional Encoding
Add absolute absolute positional encoding. 
"""

def get_angles(position, k, d_model):
  # all values of each k
  angle = 1 / np.power(10000, 2 * (k // 2) / d_model)
  # matrix multiplied into all positions - represent each position with a d_model sized vector
  return position @ angle

def abs_positional_encoding(max_position, d_model, n=3):
  """Returns absolute position encoding, creating a vector representation for all positions
  from 0 to max_position of shape (d_model,) -> a matrix of shape (max_position, d_model)
  and broadcasts it to n dimensions
  """
  # angles are of shape (positions, d_model)
  angles = get_angles(np.arange(max_position)[:, np.newaxis], 
                      np.arange(d_model)[np.newaxis, :], 
                      d_model)
  
  # apply sin to the even indices along the last axis
  angles[:, 0::2] = np.sin(angles[:, 0::2])

  # apply cos to the odd indices along the last axis
  angles[:, 1::2] = np.cos(angles[:, 1::2])

  # broadcast to n dimensions
  for _ in range(n - 2):
    angles = angles[np.newaxis, :]
  return tf.cast(angles, tf.float32)

"""##Masking Strategies
"""

def create_padding_mask(seq, n=4):
  """
  Creates padding mask for a batch of sequences seq. Mask will be of shape
  (batch_size, seq_len), and can be broadcasted to n dimensions
  """
  mask = tf.cast(tf.equal(seq, 0), tf.float32) # mask is 1 where seq is 0
  # tf.cast() converts the given tensor type into a new type of tensor e.g. tf.bool => tf.float32
  # tf.equal(x,y) returns True where i'th element in the x equals to y and False when isn't equal to y.

  # reshape to # batch_size, 1, ..., 1. seq_len
  return tf.reshape(mask, (tf.shape(mask)[0], *[1 for _ in range(n-2)], tf.shape(mask)[-1]))

def create_look_ahead_mask(seq_len):
  """
  Creates an upper triangular mask of ones of shape (seq_len seq_len).
  It is the same for all inputs of shape seq_len
  """
  mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  return tf.cast(mask, tf.float32) # (seq_len, seq_len)

def create_mask(inp, n=4):
  """Function to create the proper mask for an input batch
  mask = max(padding_mask, look_ahead_mask)

  Args:
    inp: batch tensor of input sequences of shape (..., seq_len)
  """
  padding_mask = create_padding_mask(inp, n)
  look_ahead_mask = create_look_ahead_mask(inp.shape[-1])

  # create final mask
  return tf.maximum(padding_mask, look_ahead_mask)

"""## Skewing Algorithm
Skewing algorithm to order the queries.
"""

def skew(t: tf.Tensor):
  """
  Implements skewing algorithm given by Huang et. al 2018 to reorder the
  dot(Q, RelativePositionEmbeddings) matrix into the correct ordering for which
  Tij = compatibility of ith query in Q with relative position (j - i)

  This implementation accounts for rank n tensors

  Algorithm:
      1. Pad T
      2. Reshape
      3. Slice

  T is supposed to be of shape (..., L, L), but the function generalizes to any shape
  """
  # pad the input tensor
  middle_paddings = [[0, 0] for _ in range(len(t.shape) - 1)]
  padded = tf.pad(t, [*middle_paddings, [1, 0]])

  # reshape
  Srel = tf.reshape(padded, (-1, t.shape[-1] + 1, t.shape[-2]))
  Srel = Srel[:, 1:] # slice required positions
  return tf.cast(tf.reshape(Srel, t.shape), t.dtype)

"""## Relatived Scale Dot Product Attention
Finally the main equations implementation. 
"""

def rel_scaled_dot_prod_attention(q, k, v, e, mask=None):
  """
  Implements equation 3 given in the previous section to calculate the attention weights,
  Mask has different shapes depending on its type (padding, look_ahead or combined),
  but by scaling and adding it to the attention logits, masking can be performed

  Attention = softmax(mask(QKT + skew(QET))/sqrt(d_k))V

  Args:
    q: Queries matrix of shape (..., seq_len_q, d_model)
    k: Keys matrix of shape (..., seq_len_k, d_model)
    v: Values matrix of shape (..., seq_len_k, d_model)
    e: Relative Position embedding matrix of shape (seq_len_k, d_model)
  
  Returns:
    output attention, attention weights
  """
  QKt = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)
  Srel = skew(tf.matmul(q, e, transpose_b=True)) # (..., seq_len_q, seq_len_k)

  # calculate and scale logits
  dk = math.sqrt(k.shape[-1]) 
  scaled_attention_logits = (QKt + Srel) / dk

  # add the mask to the attention logits
  if mask is not None:
    scaled_attention_logits += (mask * -1e09) # mask is added only to attention logits
  
  # softmax is normalized on the last axis so that the ith row adds up to 1
  # this is best for multiplication by v because the last axis (made into 
  # probabilities) interacts with the values v
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v) # (..., seq_len_q, d_k)
  return output, attention_weights

"""## Multi-Head Attention

###Some Helper Functions
"""

# helper function
def split_heads(x, num_heads, depth=None):
  """
  assumes x is of shape (..., num_heads * depth)
  split the last dimension of x into (num_heads, depth),
  transposes to (..., num_heads, L, depth)
  """
  if depth is None:
    assert x.shape[-1] % num_heads == 0
    depth = x.shape[-1] // num_heads

  # split d_model into h, d_h
  x = tf.reshape(x, (*x.shape[:-1], num_heads, depth)) # (..., L, num_heads, depth)

  # transpose axes -2 and -3 - tf specifies this with perm so all this fluff needs to be done
  final_perm = len(x.shape) - 1
  prior_perms = np.arange(0, final_perm - 2) # axes before the ones that need to be transposed

  # transpose to shape (..., num_heads, L, depth)
  return tf.transpose(x, perm=[*prior_perms, final_perm-1, final_perm-2, final_perm])

# another helper function
def get_required_embeddings(E, seq_len, max_len=None):
  """
  Given an input sequence of length seq_len, which does not necessary equal max_len, the 
  maximum relative distance the model is set to handle, embeddings in E from the right are 
  the required relative positional embeddings
  Embeddings have to be taken from the right because E is considered to be 
  ordered from -max_len + 1 to 0
  For all positions distanced past -max_len + 1, use E_{-max_len + 1}
  """
  if not E.built:
    E.build(seq_len)
  if max_len is None:
    max_len = E.embeddings.get_shape()[0] # assumes E is a keras.layers.Embedding
  
  if max_len >= seq_len:
    seq_len = min(seq_len, max_len)
    return E(np.arange(max_len - seq_len, max_len))
  
  return tf.concat(
      values=[*[E(np.arange(0, 1)) for _ in range(seq_len - max_len)], E(np.arange(0, max_len))], 
      axis=0
  )

"""### Multi-Head Attention Block"""

#MAX_LENGTH = CONFIG.MAX_LENGTH
#MAX_REL_DIST = CONFIG.MAX_REL_DIST

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, max_rel_dist, use_bias=True):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.max_len = max_rel_dist

    assert d_model % num_heads == 0, "d_model must be divisible into num_heads"

    self.depth = self.d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model, use_bias=use_bias) # parameter matrix to generate Q from input
    self.wk = tf.keras.layers.Dense(d_model, use_bias=use_bias) # parameter matrix to generate K from input
    self.wv = tf.keras.layers.Dense(d_model, use_bias=use_bias) # parameter matrix to generate V from input

    self.E = tf.keras.layers.Embedding(self.max_len, self.d_model) # relative position embeddings

    self.wo = tf.keras.layers.Dense(d_model, use_bias=use_bias) # final output parameter matrix
 
  def call(self, q, k, v, mask=None):
    """
    Creates Q, K, and V, gets required embeddings in E, splits into heads,
    computes attention, concatenates, and passes through final output layer
    """
    # Get Q, K, V
    q = self.wq(q) # (batch_size, seq_len, d_model)
    k = self.wk(k) # (batch_size, seq_len, d_model)
    v = self.wv(v) # (batch_size, seq_len, d_model)
    
    # Get E
    seq_len_k = k.shape[-2]
    e = get_required_embeddings(self.E, seq_len_k, self.max_len) # (seq_len_k, d_model)

    # split into heads
    q = split_heads(q, self.num_heads, self.depth) # (batch_size, h, seq_len_q, depth)
    k = split_heads(k, self.num_heads, self.depth) # (batch_size, h, seq_len_k, depth)
    v = split_heads(v, self.num_heads, self.depth) # (batch_size, h, seq_len_k, depth)
    e = split_heads(e, self.num_heads, self.depth) # (            h, seq_len_k, depth)

    # rel_scaled_attention shape = (batch_size, h, seq_len_q, depth)
    # attention_weights shape = (batch_size, h, seq_len_q, seq_len_k)
    rel_scaled_attention, attention_weights = rel_scaled_dot_prod_attention(q, k, v, e, mask=mask)

    # transpose rel_scaled_attention back to (batch_size seq_len_q, h, depth)
    final_perm = len(rel_scaled_attention.shape) - 1 # can't use rank for some reason
    prior_perms = np.arange(0, final_perm - 2) # axes before the ones that need to be transposed
    rel_scaled_attention = tf.transpose(rel_scaled_attention,
                                        perm=[*prior_perms, final_perm-1, final_perm-2, final_perm])

    # concatenate heads -> (batch_size, seq_len, d_model)
    sh = rel_scaled_attention.shape
    concat_attention = tf.reshape(rel_scaled_attention, (*sh[:-2], self.d_model)) 

    output = self.wo(concat_attention)

    return output, attention_weights

"""##Pointwise Feed Forward Network"""

class PointwiseFFN(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, use_bias=True):
    super(PointwiseFFN, self).__init__()

    self.main = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', use_bias=use_bias), # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, use_bias=use_bias) # (batch_size, seq_len, d_model)           
    ])
  
  def call(self, x):
    return self.main(x)

"""## Decoder&TransformerDecoder Layers"""

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, max_rel_dist, 
               use_bias=True, dropout=0.1, layernorm_eps=1e-06):
    super(DecoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads, max_rel_dist=max_rel_dist, use_bias=use_bias)
    self.ffn = PointwiseFFN(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=layernorm_eps)
    self.layernorm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=layernorm_eps)

    self.dropout1 = tf.keras.layers.Dropout(dropout)
    self.dropout2 = tf.keras.layers.Dropout(dropout)
  
  def call(self, x, training=False, mask=None):
    attn_output, attn_weights = self.mha(x, x, x, mask=mask) # calculate attention
    attn_output = self.dropout1(attn_output, training=training) # dropout
    # layernorm on residual connection
    out1 = self.layernorm1(x + attn_output) # (batch_size, seq_len, d_model)

    ffn_output = self.ffn(out1) # pass through FFN
    ffn_output = self.dropout2(ffn_output, training=training) # dropout
    # layernorm on residual connection
    out2 = self.layernorm2(out1 + ffn_output) # (batch_size, seq_len, d_model)

    return out2, attn_weights

class TransformerDecoder(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_rel_dist, 
               max_abs_position=20000, use_bias=True, dropout=0.1, layernorm_eps=1e-06, tie_emb=False):
    super(TransformerDecoder, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.tie_emb = tie_emb
    self.le = layernorm_eps

    self.max_position = max_abs_position # might need for decode

    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model) # input embeddings
    self.positional_encoding = abs_positional_encoding(max_abs_position, d_model) # absolute position encoding
    self.dropout = tf.keras.layers.Dropout(dropout) # embedding dropout

    # decoder layers
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, max_rel_dist, use_bias, dropout, layernorm_eps)\
                       for _ in range(self.num_layers)]

    # final layer is linear or embedding weight depending on tie emb
    if not tie_emb:
      self.final_layer = tf.keras.layers.Dense(vocab_size, use_bias=use_bias)
  
  def call(self, x, training=False, mask=None):
    # initialize attention weights dict to output
    attention_weights = {}

    # embed x and add absolute positional encoding
    x = self.embedding(x) # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
    x *= math.sqrt(self.d_model) 
    if self.max_position > 0:
      x += self.positional_encoding[:, :x.shape[-2], :]

    x = self.dropout(x, training=training)

    # pass through decoder layers
    for i in range(len(self.dec_layers)):
      x, w_attn = self.dec_layers[i](x, training, mask)
      attention_weights[f'DecoderLayer{i+1}'] = w_attn
    
    # final layer
    if self.tie_emb:
      x = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
    else:
      x = self.final_layer(x)
    
    # returns unsoftmaxed logits
    return x, attention_weights

"""##Custom Learning Rate Schedule"""

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)