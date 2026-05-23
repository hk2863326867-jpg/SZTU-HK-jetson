import tensorflow as tf

def dense(inputs, units, activation=None, name=None):
    """兼容的全连接层"""
    # 使用tf.keras.layers.Dense替代tf.compat.v1.layers.dense，以兼容Keras 3.0
    return tf.keras.layers.Dense(
        units=units,
        activation=activation,
        name=name
    )(inputs)

def batch_normalization(inputs, name=None):
    """兼容的批归一化层"""
    # 使用tf.keras.layers.BatchNormalization替代tf.compat.v1.layers.batch_normalization，以兼容Keras 3.0
    return tf.keras.layers.BatchNormalization(
        name=name
    )(inputs)
