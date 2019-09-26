import tensorflow as tf

from base import BaseModel
from models.elmo_layer import ElmoEmbeddingLayer


class BlstmModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build_model()

    def build_model(self):
        vocab_size = int(self.config.model.vocab_size)
        embedding_size = int(self.config.model.embedding_size)
        lstm_units = int(self.config.model.lstm_units)
        output_size = int(self.config.model.output_size)
        batch_size = int(self.config.trainer.batch_size)
        use_elmo = bool(self.config.model.use_elmo)

        # input layer
        input_dtype = 'string' if use_elmo else None
        _input = tf.keras.layers.Input(shape=(None,), batch_size=batch_size, dtype=input_dtype)

        # embeddings layer
        if use_elmo:
            embeddings = ElmoEmbeddingLayer()(_input)
            embedding_size = 1024  # hard coded in elmo
        else:
            embeddings = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)(_input)

        bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                                 input_shape=(batch_size, None, embedding_size)),
            merge_mode='sum'
        )(embeddings)

        logits = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_size))(bilstm)

        mask = tf.keras.layers.Input(shape=(None, output_size), batch_size=batch_size)
        masked_logits = tf.keras.layers.Add()([logits, mask])
        output = tf.keras.layers.Softmax()(masked_logits)

        self.model = tf.keras.Model(inputs=[_input, mask], outputs=output, name='bi-lstm')

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc']
        )
