import tensorflow as tf
import tensorflow_hub as hub


class ElmoEmbeddingLayer(tf.keras.layers.Layer):
    # source: https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
    # http://sujayskumar.com/2018/10/02/elmo-embeddings-in-keras/
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.prepare_session()

        self._trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        r = self.elmo(tf.keras.backend.squeeze(tf.keras.backend.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['elmo']
        result = r
        return result

    def compute_mask(self, inputs, mask=None):
        return tf.keras.backend.not_equal(inputs, '__PAD__')

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.dimensions

    @staticmethod
    def prepare_session():
        # https://github.com/tensorflow/hub/blob/master/docs/common_issues.md#running-inference-on-a-pre-initialized-module
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)

        tf.keras.backend.set_session(sess)
        # Initialize sessions
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
