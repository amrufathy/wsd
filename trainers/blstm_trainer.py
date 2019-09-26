import os

import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from base import BaseTrainer


class SaveModel(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.filepath = file_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath)


class BlstmTrainer(BaseTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.callbacks = []

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.callbacks.earlystopping_monitor,
                patience=self.config.callbacks.earlystopping_patience,
                restore_best_weights=self.config.callbacks.earlystopping_restore_best_weights
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        self.callbacks.append(
            SaveModel(
                file_path=os.path.join(self.config.callbacks.checkpoint_dir, 'ckpt-model-{epoch:02d}.h5'),
            )
        )

    def train(self, data):
        batch_size = self.config.trainer.batch_size

        history = self.model.fit_generator(
            self.batch_generator(data),
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=self.config.trainer.examples_size // batch_size,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks,
        )

    def batch_generator(self, data, shuffle=True):
        """
        Generates a batch of equally padded examples and labels.

        Padding is computed per batch to support variable sentences length
            during train.
        """
        examples, labels, mask = data
        sense_labels, pos_labels, lex_labels = labels
        batch_size = self.config.trainer.batch_size
        use_elmo = bool(self.config.model.use_elmo)
        n_classes = self.config.model.output_size

        if not shuffle:
            perm = np.array(range(len(examples)))
        else:
            perm = np.random.permutation(len(examples))

        start = 0
        while True:  # loop infinite times over the dataset
            end = start + batch_size

            examples_batch = examples[perm[start:end]]

            sense_labels_batch = sense_labels[perm[start:end]]
            pos_labels_batch = pos_labels[perm[start:end]]
            lex_labels_batch = lex_labels[perm[start:end]]

            mask_batch = mask[perm[start:end]]

            if use_elmo:
                examples_batch = pad_sequences(examples_batch, padding='post', value='__PAD__', dtype=object).tolist()
                examples_batch = [' '.join(t) for t in examples_batch]
                examples_batch = np.expand_dims(examples_batch, -1)
            else:
                examples_batch = pad_sequences(examples_batch, padding='post')

            sense_labels_batch = np.expand_dims(pad_sequences(sense_labels_batch, padding='post'), -1)
            pos_labels_batch = np.expand_dims(pad_sequences(pos_labels_batch, padding='post'), -1)
            lex_labels_batch = np.expand_dims(pad_sequences(lex_labels_batch, padding='post'), -1)

            # by default don't pass any thing
            mask_array = np.full(shape=(*sense_labels_batch.squeeze().shape, n_classes), fill_value=-np.inf)

            for i in range(len(mask_array)):
                # each sent
                for j in range(len(mask_array[i])):
                    # each word, get senses ids + random ids
                    if j < len(mask_batch[i]):
                        rands = np.random.randint(n_classes, size=10)
                        idx = np.append(mask_batch[i][j], rands)
                        mask_array[i][j][idx] = 0.
                    else:  # padding
                        mask_array[i][j][0] = 0.

            # https://github.com/tensorflow/tensorflow/issues/26699#issuecomment-473370624
            yield [examples_batch, mask_array], [sense_labels_batch, pos_labels_batch, lex_labels_batch]

            if start + batch_size >= len(examples):
                # reset
                start = 0
                if not shuffle:
                    perm = np.array(range(len(examples)))
                else:
                    perm = np.random.permutation(len(examples))
            else:
                start += batch_size
