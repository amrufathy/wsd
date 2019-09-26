import tensorflow as tf

from base import BaseModel
from models.attention import Attention
from models.elmo_layer import ElmoEmbeddingLayer


class MultiTaskModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build_model()

    def build_model(self):
        vocab_size = int(self.config.model.vocab_size)
        embedding_size = int(self.config.model.embedding_size)
        lstm_units = int(self.config.model.lstm_units)
        output_size = int(self.config.model.output_size)
        pos_size = int(self.config.model.pos_size)
        lex_size = int(self.config.model.lex_size)
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

        bilstm, forward_h, _, backward_h, _ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True, dropout=0.2,
                                 recurrent_dropout=0.2, input_shape=(batch_size, None, embedding_size)),
            merge_mode='sum'
        )(embeddings)

        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])

        ctx, attn = Attention(lstm_units)([bilstm, state_h])

        conc = tf.keras.layers.Concatenate()([bilstm, ctx])

        # decoder
        dec_lstm = tf.keras.layers.LSTM(
            lstm_units, return_sequences=True, input_shape=(None, None, 1024),
            dropout=0.2, recurrent_dropout=0.2
        )(conc)

        wsd_logits = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_size), name='wsd_logits')(dec_lstm)

        wsd_mask = tf.keras.layers.Input(shape=(None, output_size), batch_size=batch_size, name='wsd_mask')
        masked_wsd_logits = tf.keras.layers.Add(name='masked_wsd_logits')([wsd_logits, wsd_mask])

        wsd_output = tf.keras.layers.Softmax(name='wsd')(masked_wsd_logits)

        pos_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(pos_size, activation='softmax'), name='pos')(dec_lstm)

        lex_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(lex_size, activation='softmax'), name='lex')(dec_lstm)

        self.model = tf.keras.Model(inputs=[_input, wsd_mask], outputs=[wsd_output, pos_output, lex_output], name='multitask')

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc']
        )

        tf.keras.utils.plot_model(self.model, show_shapes=True)
