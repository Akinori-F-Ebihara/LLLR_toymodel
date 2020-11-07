"""LSTM model definition compatible with TensorFlow's eager execution.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation


class MLP4DRE(tf.keras.Model):
    
    def __init__(self, nb_cls, feat_dim, activation='relu'):
        """
        Args:
            nb_cls: An int. The dimension of the output logit vectors.
            feat_dim: An int. Dimensions of 3 FC layers.
        """
        super(MLP4DRE, self).__init__(name="MLP4DRE")

        # Parameters
        self.nb_cls = nb_cls
        self.activation = activation
        
        # Logit generation with fully-connected layer
        self.bn_logit = BatchNormalization()
        self.activation_logit = Activation(self.activation)
        self.fc1 = Dense(feat_dim, activation=None, use_bias=False)
        self.fc2 = Dense(feat_dim, activation=None, use_bias=False)
        self.fc_logit = Dense(nb_cls, activation=None, use_bias=False)
    

    def call(self, inputs, training):
        """Calc logits.
        Args:
            inputs: A Tensor with shape=(batch, feature dimension). E.g. (1000, 100).
            training: A boolean. Training flag used in BatchNormalization and dropout.
        Returns:
            outputs: A Tensor with shape=(batch, duration, nb_cls).
        """
        
        outputs = self.fc1(inputs)
        outputs = self.bn_logit(outputs, training=training)
        outputs = self.activation_logit(outputs)
        
        outputs = self.fc2(outputs)
        outputs = self.bn_logit(outputs, training=training)
        outputs = self.activation_logit(outputs)
        
        outputs = self.fc_logit(outputs)
        
        return outputs