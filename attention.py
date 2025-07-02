import os

from keras import backend as K
from keras.layers import Dense, Lambda, Dot, Activation, Concatenate, Layer

# KERAS_ATTENTION_DEBUG: If set to 1. Will switch to debug mode.
# In debug mode, the class Attention is no longer a Keras layer.
# What it means in practice is that we can have access to the internal values
# of each tensor. If we don't use debug, Keras treats the object
# as a layer and we can only get the final output.
debug_flag = int(os.environ.get('KERAS_ATTENTION_DEBUG', 0))


class Attention(object if debug_flag else Layer):

    def __init__(self, units=128, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope(self.name if not debug_flag else 'attention'):
            self.attention_score_vec = Dense(input_dim, use_bias=False, name='attention_score_vec')
            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
            self.attention_score = Dot(axes=[1, 2], name='attention_score')
            self.attention_weight = Activation('softmax', name='attention_weight')
            self.context_vector = Dot(axes=[1, 1], name='context_vector')
            self.attention_output = Concatenate(name='attention_output')
            self.attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        if not debug_flag:
            # debug: the call to build() is done in call().
            super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            return self.call(inputs, training, **kwargs)
        else:
            return super(Attention, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @param training: not used in this layer.
        @return: 2D tensor with shape (batch_size, units)
        @author: felixhao28, philipperemy.
        """
        if debug_flag:
            self.build(inputs.shape)
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = self.attention_score_vec(inputs)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(Attention, self).get_config()
        config.update({'units': self.units})
        return config

from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as K


class FeatureAttention(Layer):
    def __init__(self, num_features, hidden_units, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
        self.num_features = num_features
        self.hidden_units = hidden_units

    def build(self, input_shape):
        # Dense layer to compute attention weights over features
        self.feature_attention = Dense(self.num_features, activation="softmax", name="feature_attention")
        # Dense layer to map hidden units to features
        self.hidden_to_features = Dense(self.num_features, activation="linear", name="hidden_to_features")
        super(FeatureAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: (batch_size, sequence_length, hidden_units)

        # Compute feature-level attention weights
        attention_weights = self.feature_attention(inputs)  # Shape: (batch_size, sequence_length, num_features)

        # Project hidden units to feature space
        projected_inputs = self.hidden_to_features(inputs)  # Shape: (batch_size, sequence_length, num_features)

        # Apply attention weights to the projected inputs
        weighted_features = attention_weights * projected_inputs  # Shape: (batch_size, sequence_length, num_features)

        # Compute context vector by summing over features
        context_vector = K.sum(weighted_features, axis=-1)  # Shape: (batch_size, sequence_length)

        return context_vector, attention_weights

    def get_config(self):
        config = super(FeatureAttention, self).get_config()
        config.update({"num_features": self.num_features, "hidden_units": self.hidden_units})
        return config
