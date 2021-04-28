import tensorflow as tf


class KerasModel(tf.keras.models.Model):
    def get_config(self):
        pass

    def __init__(self):
        super(KerasModel, self).__init__()

    def call(self, inputs, training=None, mask=None):
        return inputs


class TFModel(tf.Module):
    def __init__(self):
        super(TFModel, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int8),
                                  tf.TensorSpec(shape=(), dtype=tf.int8)])
    def add(self, a, b):
        return tf.add(a, b)

    def __call__(self, *args, **kwargs):
        pass
