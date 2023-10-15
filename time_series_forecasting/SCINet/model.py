import tensorflow as tf
import numpy as np
import os
import math
import random
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

tfk = tf.keras
tfkl = tf.keras.layers

from typing import Tuple
from tensorflow.keras.regularizers import L1L2


class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: float, kernel_size: int, neg_slope: float = .01, dropout: float = .5, **kwargs):
        if filters <= 0 or h <= 0:
            raise ValueError('filters and h must be positive')
        super(InnerConv1DBlock, self).__init__(**kwargs)
        self.conv1d = tf.keras.layers.Conv1D(max(round(h * filters), 1), kernel_size, padding='same')
        self.leakyrelu = tf.keras.layers.LeakyReLU(neg_slope)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv1d2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.tanh = tf.keras.activations.tanh

    def call(self, input_tensor, training=None):
        x = self.conv1d(input_tensor)
        x = self.leakyrelu(x)

        if training:
            x = self.dropout(x)

        x = self.conv1d2(x)
        x = self.tanh(x)
        return x


class SciBlock(tf.keras.layers.Layer):
    def __init__(self, features: int, kernel_size: int, h: int, **kwargs):
        """
        :param features: number of features in the output
        :param kernel_size: kernel size of the convolutional layers
        :param h: scaling factor for convolutional module
        """

        super(SciBlock, self).__init__(**kwargs)
        self.features = features
        self.kernel_size = kernel_size
        self.h = h

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'features': self.features,
            'kernel_size': self.kernel_size,
            'h': self.h,
        })
        return config

    def build(self, input_shape):
        self.conv1ds = {k: InnerConv1DBlock(filters=self.features, h=self.h, kernel_size=self.kernel_size, name=k)
                        for k in ['psi', 'phi', 'eta', 'rho']}  # regularize?
        super().build(input_shape)
        # [layer.build(input_shape) for layer in self.conv1ds.values()]  # unneeded?

    def call(self, inputs, training=None):
        F_odd, F_even = inputs[:, ::2], inputs[:, 1::2]

        # Interactive learning as described in the paper
        F_s_odd = F_odd * tf.math.exp(self.conv1ds['phi'](F_even))
        F_s_even = F_even * tf.math.exp(self.conv1ds['psi'](F_odd))

        F_prime_odd = F_s_odd + self.conv1ds['rho'](F_s_even)
        F_prime_even = F_s_even - self.conv1ds['eta'](F_s_odd)

        return F_prime_odd, F_prime_even


class Interleave(tf.keras.layers.Layer):
    """A layer used to reverse the even-odd split operation."""

    def __init__(self, **kwargs):
        super(Interleave, self).__init__(**kwargs)

    def interleave(self, slices):
        if not slices:
            return slices
        elif len(slices) == 1:
            return slices[0]

        mid = len(slices) // 2
        even = self.interleave(slices[:mid])
        odd = self.interleave(slices[mid:])

        shape = tf.shape(even)
        return tf.reshape(tf.stack([even, odd], axis=3), (shape[0], shape[1] * 2, shape[2]))

    def call(self, inputs):
        return self.interleave(inputs)


class SciNet(tf.keras.layers.Layer):
    def __init__(self, horizon: int, features: int, levels: int, h: int, kernel_size: int,
                 regularizer: Tuple[float, float] = (0, 0), **kwargs):
        """
        :param horizon: number of time stamps in output
        :param features: number of features in output
        :param levels: height of the binary tree + 1
        :param h: scaling factor for convolutional module in each SciBlock
        :param kernel_size: kernel size of convolutional module in each SciBlock
        :param regularizer: activity regularization (not implemented)
        """

        if levels < 1:
            raise ValueError('Must have at least 1 level')
        super(SciNet, self).__init__(**kwargs)
        self.horizon = horizon
        self.features = features
        self.levels = levels
        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            horizon * features,
            kernel_regularizer=L1L2(0.001, 0.01),
            # activity_regularizer=L1L2(0.001, 0.01)
        )
        # self.regularizer = tf.keras.layers.ActivityRegularization(l1=regularizer[0], l2=regularizer[1])

        # tree of sciblocks
        self.sciblocks = [SciBlock(features=features, kernel_size=kernel_size, h=h)
                          for _ in range(2 ** levels - 1)]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'horizon': self.horizon,
            'features': self.features,
            'levels': self.levels,
            'interleave': self.interleave,
            'flatten': self.flatten,
            'dense': self.dense,
            'sciblocks': self.sciblocks,
        })
        return config

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} must be evenly divisible by a tree with '
                             f'{self.levels} levels')
        super().build(input_shape)
        # [layer.build(input_shape) for layer in self.sciblocks]  # input_shape

    def call(self, inputs, training=None):
        # cascade input down a binary tree of sci-blocks
        lvl_inputs = [inputs]  # inputs for current level of the tree
        for i in range(self.levels):
            i_end = 2 ** (i + 1) - 1
            i_start = i_end - 2 ** i
            lvl_outputs = [output for j, tensor in zip(range(i_start, i_end), lvl_inputs)
                           for output in self.sciblocks[j](tensor)]
            lvl_inputs = lvl_outputs

        x = self.interleave(lvl_outputs)
        x += inputs

        # not sure if this is the correct way of doing it. The paper merely said to use a fully connected layer to
        # produce an output. Can't use TimeDistributed wrapper. It would force the layer's timestamps to match that of
        # the input -- something SCINet is supposed to solve
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.reshape(x, (-1, self.horizon, self.features))

        return x


class StackedSciNet(tf.keras.layers.Layer):
    def __init__(self, horizon: int, features: int, stacks: int, levels: int, h: int, kernel_size: int,
                 regularizer: Tuple[float, float] = (0, 0), **kwargs):
        """
        :param horizon: number of time stamps in output
        :param stacks: number of stacked SciNets
        :param levels: number of levels for each SciNet
        :param h: scaling factor for convolutional module in each SciBlock
        :param kernel_size: kernel size of convolutional module in each SciBlock
        :param regularizer: activity regularization (not implemented)
        """

        if stacks < 1:
            raise ValueError('Must have at least 1 stack')
        super(StackedSciNet, self).__init__(**kwargs)
        self.horizon = horizon
        self.scinets = [SciNet(horizon=horizon, features=features, levels=levels, h=h, kernel_size=kernel_size,
                               regularizer=regularizer) for _ in range(stacks)]
        self.mse_fn = tf.keras.metrics.MeanSquaredError()
        self.mae_fn = tf.keras.metrics.MeanAbsoluteError()

    # def build(self, input_shape):
    #     super().build(input_shape)
    #     [stack.build(input_shape) for stack in self.scinets]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'horizon': self.horizon,
            'features': self.features,
            'stacks': self.stacks,
            'levels': self.levels,
            'h': self.h,
            'kernel_size': self.kernel_size,
            'regularizer': self.regularizer,
        })
        return config

    def call(self, inputs, targets=None, sample_weights=None, training=None):
        outputs = []
        for scinet in self.scinets:
            x = scinet(inputs)
            outputs.append(x)  # keep each stack's output for intermediate supervision
            inputs = tf.concat([x, inputs[:, x.shape[1]:, :]], axis=1)

        if targets is not None:
            # Calculate metrics
            mse = self.mse_fn(targets, x, sample_weights)
            mae = self.mae_fn(targets, x, sample_weights)
            self.add_metric(mse, name='mean_squared_error')
            self.add_metric(mae, name='mean_absolute_error')

            if training:
                # Calculate loss as sum of mean of norms of differences between output and input feature vectors for
                # each stack
                stacked_outputs = tf.stack(outputs)
                differences = stacked_outputs - targets
                loss = tf.linalg.normalize(differences, axis=1)[1]
                loss = tf.reshape(loss, (-1, self.horizon))
                loss = tf.reduce_sum(loss, 1)
                loss = loss / self.horizon
                loss = tf.reduce_sum(loss)
                self.add_loss(loss)

        return x



class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/SciNetTest_downsampled_3.h5') , custom_objects={"StackedSciNet": StackedSciNet, "MeanSquaredError" : tf.keras.metrics.MeanSquaredError, "MeanAbsoluteError" : tf.keras.metrics.MeanAbsoluteError, })

    def predict(self, X):


        window = 720
        stride = 16
        output_sampling = 0.5
        final_size = 864

        to_predict = int(final_size*output_sampling+4)
        

        placeholder = np.empty([stride, 7])
        placeholder = np.expand_dims(placeholder, axis=0)

        X = {'inputs': X, 'targets': placeholder}

        X1 = X['inputs'][::2]

        col_max = np.max(X1, axis=0)
        col_min = np.min(X1, axis=0)
        X_norm = np.divide(X1 - col_min, col_max - col_min)
        X1 = X_norm[-window:]
        X1 = np.expand_dims(X1, axis=0)

        X['inputs'] = X1

        out = np.array([])
        
        for i in range(int(math.ceil(to_predict / stride))):
            pred = self.model.predict(X)
            pred = np.array(pred)

            X1 = X['inputs']

            X1 = np.concatenate((X1[0], pred[0][:stride]), axis=0)
            X1 = np.expand_dims(X1, axis=0)
            if i != 0:
                out = np.concatenate((out, pred[0][:stride]), axis=0)
            else:
                out = pred[0][:stride]


            X1 = X1[0][-window:]
            X1 = np.expand_dims(X1, axis=0)

            X['inputs'] = X1

        pred = pd.DataFrame(out)
        upsampled_pred = pd.DataFrame()

        from scipy import interpolate

        for column in pred.columns:
            y = pred[column].to_numpy()
            x = np.arange(0, len(y))
            f = interpolate.interp1d(x, y)

            x_new = np.arange(0, len(y)-1, output_sampling)
            ynew = f(x_new)
            upsampled_pred[column] = ynew

        upsampled_pred.to_numpy()


        out = upsampled_pred[:final_size]
        out = np.multiply(out, col_max - col_min)
        out = out + col_min
        out = tf.convert_to_tensor(out)
        out = tf.cast(out, tf.float32)
        return out