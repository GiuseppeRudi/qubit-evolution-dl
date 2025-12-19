import numpy as np
from ..registry import register_trainer

@register_trainer("standard")
class StandardTrainer:
    def __init__(self, model, model_cfg, eval_cfg=None):
        self.model = model
        self.cfg = model_cfg
        self.eval_cfg = eval_cfg or {}

    def fit(self, splits):
        return self.model.fit(
            splits.X_train, splits.Y_train,
            epochs=self.cfg.training.epochs,
            batch_size=self.cfg.training.batch_size,
            validation_data=(splits.X_val, splits.Y_val),
            verbose=1,
        )

    def predict_sample(self, splits):
        idx = int(self.eval_cfg.get("sample_index", 0))
        sample_x = splits.X_train[idx:idx+1]
        pred = self.model.predict(sample_x, verbose=0)
        return sample_x, pred

    def report_sample(self, sample_x, pred):
        steps = int(self.eval_cfg.get("print_steps", 5))
        print(f"  X shape: {sample_x.shape}")
        print(f"  Pred shape: {pred.shape}")
        np.set_printoptions(suppress=True, precision=16)
        print(pred[0, :steps, :])
