import os
import time
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import numpy as np

GRIDS = {16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8), 256: (16, 16),
         512: (32, 16), 1024: (32, 32), 2048: (64, 32)}


class W2L:
    def __init__(self, model_dir, vocab_size, n_channels, data_format):
        if data_format not in ["channels_first", "channels_last"]:
            raise ValueError("Invalid data type specified: {}. Use either "
                             "channels_first or "
                             "channels_last.".format(data_format))

        self.model_dir = model_dir
        self.data_format = data_format
        self.cf = self.data_format == "channels_first"
        self.n_channels = n_channels
        self.vocab_size = vocab_size
        self.hidden_dim = 16  # TODO don't hardcode

        if os.path.isdir(model_dir) and os.listdir(model_dir):
            print("Model directory already exists. Loading last model...")
            last = self.get_last_model()
            self.model = tf.keras.models.load_model(
                os.path.join(model_dir, last),
                custom_objects={"Conv1DTranspose": Conv1DTranspose})
            self.step = int(last[:-3])
            print("...loaded {}.".format(last))
        else:
            print("Model directory does not exist. Creating new model...")
            self.model = self.make_w2l_model()
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            self.step = 0

        self.writer = tf.summary.create_file_writer(model_dir)

    def make_w2l_model(self):
        """Creates a Keras model that does the W2L forward computation.

        Just goes from mel spectrogram input to logits output.

        Returns:
            Keras sequential model.

        TODO could allow model configs etc. For now, architecture is hardcoded

        """
        channel_ax = 1 if self.cf else -1

        def conv1d(n_f, w_f, stride):
            return layers.Conv1D(
                n_f, w_f, stride, padding="same", data_format=self.data_format,
                use_bias=False)

        def conv1d_t(n_f, w_f, stride):
            return Conv1DTranspose(
                n_f, w_f, stride, padding="same", data_format=self.data_format,
                use_bias=False)

        def act():
            return layers.ReLU()

        layer_list_enc = [
            layers.BatchNormalization(channel_ax),
            conv1d(256, 48, 2),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(2048, 32, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d(2048, 1, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            layers.Conv1D(self.hidden_dim, 1, 1, padding="same",
                          data_format=self.data_format)
        ]

        layer_list_dec = [
            layers.BatchNormalization(channel_ax),
            conv1d_t(2048, 1, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(2048, 1, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 32, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            conv1d_t(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            act(),
            Conv1DTranspose(128, 48, 2, padding="same",
                            data_format=self.data_format)
        ]

        # w2l = tf.keras.Sequential(layer_list, name="w2l")

        inp = tf.keras.Input((self.n_channels, None) if self.cf
                             else (None, self.n_channels))
        layer_outputs_enc = [inp]
        for layer in layer_list_enc:
            layer_outputs_enc.append(layer(layer_outputs_enc[-1]))
        layer_outputs_dec = [layer_outputs_enc[-1]]
        for layer in layer_list_dec:
            layer_outputs_dec.append(layer(layer_outputs_dec[-1]))

        # only include relu layers in outputs
        relevant = layer_outputs_enc[4::3] + [layer_outputs_enc[-1]]
        relevant += layer_outputs_dec[4::3] + [layer_outputs_dec[-1]]

        w2l = tf.keras.Model(inputs=inp, outputs=relevant)

        return w2l

    def forward(self, audio, training=False, return_all=False):
        """Simple forward pass of a W2L model to compute logits.

        Parameters:
            audio: Tensor of mel spectrograms, channels_first!
            training: Bool, if true assuming training mode otherwise inference.
                      Important for batchnorm to work properly.
            return_all: Bool, if true, return list of all layer activations
                        (post-relu), with the logits at the very end.

        Returns:
            Result of applying model to audio (list or tensor depending on
            return_all).

        """
        if not self.cf:
            audio = tf.transpose(audio, [0, 2, 1])

        out = self.model(audio, training=training)
        if return_all:
            return out
        else:
            return out[-1]

    def train_step(self, audio, audio_length, optimizer, on_gpu):
        """Implements train step of the W2L model.

        Parameters:
            audio: Tensor of mel spectrograms, channels_first!
            audio_length: "True" length of each audio clip.
            optimizer: Optimizer instance to do training with.
            on_gpu: Bool, whether running on GPU. This changes how the
                    transcriptions are handled. Currently ignored!!

        Returns:
            Loss value.

        """
        with tf.GradientTape() as tape:
            recon = self.forward(audio, training=True, return_all=False)
            # after this we need logits in shape time x batch_size x vocab_size
            # TODO mask, i.e. do not compute for padding
            loss = tf.reduce_mean(tf.math.squared_difference(recon, audio))
            # audio_length = tf.cast(audio_length / 2, tf.int32)

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # probably has to go into train_full...
        # self.annealer.update_history(loss)

        return loss

    def train_full(self, dataset, steps, adam_params, on_gpu):
        """Full training logic for W2L.

        Parameters:
            dataset: tf.data.Dataset as produced in input.py.
            steps: Number of training steps.
            adam_params: List/tuple of four parameters for Adam: learning rate,
                         beta1, beta2, epsilon.
            on_gpu: Bool, whether running on a GPU.

        """
        # TODO more flexible checkpointing. this will simply do 10 checkpoints overall
        check_freq = steps // 10
        data_step_limited = dataset.take(steps)

        # TODO use annealing
        # self.annealer = AnnealIfStuck(adam_params[0], 0.1, 20000)
        # TODO don't hardcode this
        schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            [200000, 250000], [adam_params[0], adam_params[0] / 10,
                               adam_params[0] / (5 * 10)])
        opt = tf.optimizers.Adam(schedule, *adam_params[1:])
        opt.iterations.assign(self.step)

        audio_shape = [None, self.n_channels, None] if self.cf \
            else [None, None, self.n_channels]

        def train_fn(w, x):
            return self.train_step(w, x, opt, on_gpu)

        graph_train = tf.function(
            train_fn, input_signature=[tf.TensorSpec(audio_shape, tf.float32),
                                       tf.TensorSpec([None], tf.int32)])
        # graph_train = train_fn  # skip tf.function

        start = time.time()
        for features, labels in data_step_limited:
            if not self.step % check_freq:
                print("Saving checkpoint...")
                self.model.save(os.path.join(
                    self.model_dir, str(self.step).zfill(6) + ".h5"))

            loss = graph_train(features["audio"], features["length"])

            if not self.step % 500:
                stop = time.time()
                print("Step: {}. Recon: {}".format(self.step, loss.numpy()))
                print("{} seconds passed...".format(stop - start))

            if not self.step % 100:
                with self.writer.as_default():
                    tf.summary.scalar("loss/recon", loss, step=self.step)

            self.step += 1

        self.model.save(os.path.join(
            self.model_dir, str(self.step).zfill(6) + ".h5"))

    def get_last_model(self):
        ckpts = [file for file in os.listdir(self.model_dir) if
                 file.endswith(".h5")]
        if "final.h5" in ckpts:
            return "final.h5"
        else:
            return sorted(ckpts)[-1]


class AnnealIfStuck(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, factor, n_steps):
        """Anneal the learning rate if loss doesn't decrease anymore.

        Refer to
        http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html.

        Parameters:
            base_lr: LR to start with.
            factor: By what to multiply in case we're stuck.
            n_steps: How often to check if we're stuck.

        """
        super(AnnealIfStuck, self).__init__()
        self.n_steps = n_steps
        self.lr = base_lr
        self.factor = factor
        self.loss_history = tf.Variable(
            np.zeros(n_steps), trainable=False, dtype=tf.float32,
            name="loss_history")

    def __call__(self, step):
        if tf.logical_or(tf.greater(tf.math.mod(step, self.n_steps), 0),
                         tf.equal(step, 0)):
            pass
        else:
            x1 = tf.range(self.n_steps, dtype=tf.float32, name="x")
            x2 = tf.ones([self.n_steps], dtype=tf.float32, name="bias")
            x = tf.stack((x1, x2), axis=1, name="input")
            slope_bias = tf.linalg.lstsq(x, self.loss_history[:, tf.newaxis],
                                         name="solution")
            slope = slope_bias[0][0]
            bias = slope_bias[1][0]
            preds = slope * x1 + bias

            data_var = 1 / (self.n_steps - 2) * tf.reduce_sum(
                tf.square(self.loss_history - preds))
            dist_var = 12 * data_var / (self.n_steps ** 3 - self.n_steps)
            dist = tfp.distributions.Normal(slope, tf.sqrt(dist_var),
                                            name="slope_distribution")
            prob_decreasing = dist.cdf(0., name="prob_below_zero")

            if tf.less_equal(prob_decreasing, 0.5):
                self.lr *= self.factor
        return self.lr

    def check_lr(self):
        return self.lr

    def update_history(self, new_val):
        self.loss_history.assign(tf.concat((self.loss_history[1:], [new_val]),
                                           axis=0))


def dense_to_sparse(dense_tensor, sparse_val=-1):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.

    Returns:
        SparseTensor equivalent to the dense input.

    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


class Conv1DTranspose(layers.Conv1D):
    """Why does this still not exist in Keras... """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=tf.keras.activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            activity_regularizer=tf.keras.regularizers.get(
                activity_regularizer),
            kernel_constraint=tf.keras.constraints.get(kernel_constraint),
            bias_constraint=tf.keras.constraints.get(bias_constraint),
            **kwargs)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = normalize_tuple(
                self.output_padding, 1, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                                                     'greater than output padding ' +
                                     str(self.output_padding))

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 3:
            raise ValueError(
                'Inputs should have rank 3. Received input shape: ' +
                str(input_shape))
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = layers.InputSpec(ndim=3, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis = 2
        else:
            h_axis = 1

        height = inputs_shape[h_axis]
        kernel_h, = self.kernel_size
        stride_h, = self.strides

        if self.output_padding is None:
            out_pad_h = None
        else:
            out_pad_h = self.output_padding

        # Infer the dynamic output shape:
        out_height = deconv_output_length(height,
                                          kernel_h,
                                          padding=self.padding,
                                          output_padding=out_pad_h,
                                          stride=stride_h,
                                          dilation=self.dilation_rate[0])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height)
        else:
            output_shape = (batch_size, out_height, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.nn.conv1d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=convert_data_format(self.data_format, ndim=3),
            dilations=self.dilation_rate)

        if not tf.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=convert_data_format(self.data_format,
                                                ndim=3))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis = 1, 2
        else:
            c_axis, h_axis = 2, 1

        kernel_h, = self.kernel_size
        stride_h, = self.strides

        if self.output_padding is None:
            out_pad_h = None
        else:
            out_pad_h = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0])
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        return config


def normalize_tuple(value, n, name):
    """Transforms a single integer or iterable of integers into an integer tuple.
    Arguments:
      value: The value to validate and convert. Could an int, or any iterable of
        ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
    Returns:
      A tuple of n integers.
    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    'The `' + name + '` argument must be a tuple of ' +
                    str(n) + ' integers. Received: ' + str(value) + ' '
                                                                    'including element ' + str(
                        single_value) + ' of type' +
                    ' ' + str(type(single_value)))
        return value_tuple


def convert_data_format(data_format, ndim):
    if data_format == 'channels_last':
        if ndim == 3:
            return 'NWC'
        elif ndim == 4:
            return 'NHWC'
        elif ndim == 5:
            return 'NDHWC'
        else:
            raise ValueError('Input rank not supported:', ndim)
    elif data_format == 'channels_first':
        if ndim == 3:
            return 'NCW'
        elif ndim == 4:
            return 'NCHW'
        elif ndim == 5:
            return 'NCDHW'
        else:
            raise ValueError('Input rank not supported:', ndim)
    else:
        raise ValueError('Invalid data_format:', data_format)


def deconv_output_length(input_length,
                         filter_size,
                         padding,
                         output_padding=None,
                         stride=0,
                         dilation=1):
    """Determines output length of a transposed convolution given input length.
    Arguments:
        input_length: Integer.
        filter_size: Integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        output_padding: Integer, amount of padding along the output dimension.
          Can be set to `None` in which case the output length is inferred.
        stride: Integer.
        dilation: Integer.
    Returns:
        The output length (integer).
    """
    assert padding in {'same', 'valid', 'full'}
    if input_length is None:
        return None

    # Get the dilated kernel size
    filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == 'valid':
            length = input_length * stride + max(filter_size - stride, 0)
        elif padding == 'full':
            length = input_length * stride - (stride + filter_size - 2)
        elif padding == 'same':
            length = input_length * stride

    else:
        if padding == 'same':
            pad = filter_size // 2
        elif padding == 'valid':
            pad = 0
        elif padding == 'full':
            pad = filter_size - 1

        length = ((input_length - 1) * stride + filter_size - 2 * pad +
                  output_padding)
    return length
