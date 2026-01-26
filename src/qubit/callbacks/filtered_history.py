import numpy as np
import keras

class FilteredProgbar(keras.callbacks.ProgbarLogger):
    def _filtered(self, logs):
        if not logs:
            return logs

        out = {}
        for k, v in logs.items():
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                continue
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and float(v) == 0.0:
                continue
            out[k] = v
        return out

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, self._filtered(logs))

    def on_test_batch_end(self, batch, logs=None):
        super().on_test_batch_end(batch, self._filtered(logs))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, self._filtered(logs))
