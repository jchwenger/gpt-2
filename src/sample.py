import tensorflow as tf

# modified to allow for import from the main repo folder
try:
    import model
except:
    from src import model

def top_k_logits(logits, k):
    """
    Used in body() in sample_sequence() below.
    """
    # same as in cond, used to optimize performance? Isn't there a better way?
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        '''return values of max k elements for each tensor in last dim'''
        values, _ = tf.nn.top_k(logits, k=k) # returns values, indices
        min_values = values[:, -1, tf.newaxis]

        # return logits above min value, -1e10 otherwise
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def sample_sequence(*,
                    hparams,
                    length,
                    start_token=None,
                    batch_size=None,
                    context=None,
                    temperature=1,
                    top_k=0):

    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'

        # a 'vertical' tensor filled with batch_size times start_token thus we
        # can generate a batch of samples at once, all starting with the same char
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        '''Used to generate one (batch of) tokens in body below'''

        # lm_output: a dictionary with two keys, 'present', 'logits'
        lm_output = model.model(hparams=hparams,
                                X=tokens,
                                past=past,
                                reuse=tf.AUTO_REUSE) # if tf.AUTO_REUSE, we create variables if they
                                                     # do not exist, and return them otherwise

        # retrieve contents in two variable

        # logits shape: [batch, sequence, n_vocab]
        logits = lm_output['logits'][:, :, :hparams.n_vocab] # no more than current vocab?
        presents = lm_output['present']
        # set shape to: [batch_size, n_layer, 2, n_head, sequence, n_embd // n_head]
        presents.set_shape(
                    model.past_shape(hparams=hparams,
                                     batch_size=batch_size))

        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        """
        Don't feed the last context token -- leave that to the loop below
        TODO: Would be slightly faster if we called step on the entire context,
        rather than leaving the last token transformer calculation to the while loop.
        """
        # absolute first step: context[:,:-1] == []
        # context_output: dict with keys 'logits'/'presents'
        context_output = step(hparams, context[:, :-1])

        # past: context_output['present']
        # prev: context[:, -1]
        # output: context
        # (see loop_vars below)
        def body(past, prev, output):

            # generate next step: next_outputs = { 'logits', 'presents'}
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)

            # add temperature: not quite the qi = exp(zi/T)/sumj(zj/T)?
            # simplified for speed's sake? to be rummaged...
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)

            # if top_k==0 (default), just get all logits, othewise select only top
            # logits (and get -1e10 for the rest, never selected by the multinomial)
            logits = top_k_logits(logits, k=top_k)

            # sample from logits: deprecated, use tf.random.categorical instead
            # returns a vector containing the index of the sample
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)

            # returns:
            # past:
            # prev: the produced chars, which become prev
            # output: the produced chars appended to the current output
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]
                # token embedding matrix conda install -c conda-forge regex

        # all the tf.fuss to do a while True:
        # define the always true function & pass it to tf.while_loop below
        # it must accept the same arguments as body > *args
        def cond(*args):
            return True

        # the loop: desired length sets limit to iterations
        _, _, tokens = tf.while_loop(
            cond=cond,
            body=body,
            maximum_iterations=length, # the number of chars we get
            loop_vars=[ # passed to cond/body -> (past, prev, output)
                context_output['presents'],
                context[:, -1], # vector of last context tokens
                context,
            ],
            shape_invariants=[ # [batch_size, n_layer, 2, n_head, sequence, n_embd // n_head]
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens
