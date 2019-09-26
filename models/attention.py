import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    # https://androidkt.com/text-classification-using-attention-mechanism-in-keras/

    def __init__(self, units=256, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def build(self, input_shape):
        self.W1.build(input_shape[0])
        self.W2.build((input_shape[1][0], 1, input_shape[1][1]))
        self.V.build(input_shape[0])

        self.trainable_weights.append(self.W1.trainable_weights)
        self.trainable_weights.append(self.W2.trainable_weights)
        self.trainable_weights.append(self.V.trainable_weights)

        super(Attention, self).build(input_shape)

        self.built = True

    def call(self, inputs, mask=None):
        features, hidden = inputs

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features

        return tf.reshape(context_vector, tf.shape(features)), attention_weights

    def compute_mask(self, inputs, mask=None):
        return mask
