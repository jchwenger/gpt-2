import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    """
    See this paper (Gaussian Error Linear Units): https://arxiv.org/pdf/1606.08415.pdf
    And this discussion: https://datascience.stackexchange.com/questions/49522/what-is-gelu-activation
    """
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """
    Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    As a result, not only will the data be centered around 0 with stddev=1,
    but the various dimensions/features will be on the same scale (and not
    one going to 1000, the other one between 0 and 1, etc.). See:
    https://www.youtube.com/watch?v=FDCfw-YqWTE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=9
    """
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value # innermost dim

        # scale & shift factors (trainable)
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))

        # absolute mean & variance
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)

        # normalize then scale & shift
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b

        return x

def split_states(x, n):
    """
    Reshape the last dimension of x into [n, x.shape[-1]/n].
    From e.g. [[1,2,3,4],      to  [[[1,2],[3,4]],
               [5,6,7,8]]           [[5,6],[7,8]]]
    """
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """
    Smash the last two dimensions of x into a single dimension.
    (The reverse operation from above, going back to original tensor.)
    """
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    """
    x of shape e.g. [10,4,3,5], result c of shape [10,4,3,nf].
    Part of the Transformer architecture — its use in the mlp() fn below
    confirms that this is the feedforward network described here:
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks
    """
    with tf.variable_scope(scope):
        *start, nx = shape_list(x) # innermost dim

        # Weight & bias to be trained
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))

        # Reshape x & w so that compatible, then
        # x * w + b (linear)
        # Reshape result back to include external dimension of x
        c = tf.reshape(
                tf.matmul(tf.reshape(x, [-1, nx]), # flatten & keep only innermost dim
                          tf.reshape(w, [-1, nf])  # always [nx, nf]
                         )+b,
                start+[nf])

        return c

def attention_mask(nd, ns, *, dtype):
    """
    1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    In fact tf.matrix_band_part() counts from the upper left corner! (Same as np.triu/tril.)
    """
    i = tf.range(nd)[:,None] # 'vertical' vector, using None as shorthand for tf.newaxis
    j = tf.range(ns)         # 'horizontal one, that will be 'shifted to the left' (from [0,1,2] to [-3, -2,-1]
    m = i >= j - ns + nd     # check for equivalence using numpy broadcasting > matrix filled with booleans
    return tf.cast(m, dtype) # convert boolean values to 0s & 1s


def attn(x, scope, n_state, *, past, hparams):

    assert x.shape.ndims == 3            # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0 # n_state == innermost dimension of x,
                                         # required when dividing that by n_head in
                                         # split_states below

    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        # (via [batch, sequence, heads, features] thanks to split_states)
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        # transpose from [batch, heads, sequence, features] back to [batch, sequence, heads, features]
        # then merge_states: merge last two dims [heads, features] into one
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        *_, nd, ns = shape_list(w) # <3 * operator
        b = attention_mask(nd, ns, dtype=w.dtype) # tensor with 1s in the lower triangle
        b = tf.reshape(b, [1, 1, nd, ns])         # make it compatible with w by adding external dims
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)    # swap zeroes & ones (1-b), replace ones by large
                                                  # numbers 1e10, and subtract: # -1e10 will produce
        return w                                  # 0 on the softmax (applied to the returned w)


    def multihead_attn(q, k, v):
        """
        Formula: (softmax(Q*Ktranspose)/sqrt(dv))*V
        dv: scaling factor, as countereffect to dot-product attention, faster & more space-efficient than
        additive attention, growing in magnitude for larger dv values (which pushes softmax to zero)
        """
        # dot-prod, q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)

        # scaling
        dv = tf.cast(v.shape[-1].value, w.dtype) # take the innermost dim of v, then
        w = w * tf.rsqrt(dv)                     # divide w by the square root of it
                                                 # (in paper called dk == dv == dmodel//n_heads)

        # mask: ignore past positions
        w = mask_attn_weights(w)

        # softmax & dot-prod
        w = softmax(w)
        a = tf.matmul(w, v)

        return a

    with tf.variable_scope(scope):

        # linear layer, x is [batch, sequence, features], n_state == x.shape[-1].value
        c = conv1d(x, 'c_attn', n_state*3) # *3 so you can split it into 3 just below

        # split into heads for parallelized action, c is [batch, sequence, features*3]
        # q, k & v are [batch, heads, sequence, features]
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))

        # present: [k, v] in one vector, as is past
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1) # retrieve past k & v
            k = tf.concat([pk, k], axis=-2)   # add present k to pk
            v = tf.concat([pv, v], axis=-2)   # same for v to pv

        a = multihead_attn(q, k, v)

        # remerge & linear
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state) # n_state == x.shape[-1].value

        return a, present


def mlp(x, scope, n_state, *, hparams):
    """
    Transformer (inside block)s: feed-forward then nonlinearity then feed-forward
    conv1d(gelu(conv1d(x))
    """
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2

def block(x, scope, *, past, hparams):
    """
    Transformer: norm+attention then norm+linear
    """
    with tf.variable_scope(scope):

        # norm then attention
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'),
                          'attn',
                          nx,
                          past=past,
                          hparams=hparams)
        # add
        x = x + a

        # norm then linear
        m = mlp(norm(x, 'ln_2'),
                'mlp',
                nx*4,
                hparams=hparams)
        # add
        x = x + m

        return x, present

# Use * in args to prevent any positional argument being used
def past_shape(*, hparams, batch_size=None, sequence=None):
    """
    Used in sample.py, sample_sequence() to shape presents, & body() >
    while_loop to retrieve the same shape
    """
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """
    Tile (duplicate) tensor size times: from [x,y] to [[x,y],[x,y] .. size times .. [x,y]]
    Constructed so as to be able to take lists, tuples, etc. as input.
    """
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims

    # expand [x,y] to [[x,y]], then tile [size] times according to the outer dim
    # ([size] + [1]*ndims turning into e.g. [3,1,1])
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []

        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer

        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)

        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
