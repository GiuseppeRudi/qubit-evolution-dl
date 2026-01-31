import math
import numpy as np
import optuna
import tensorflow as tf
import re


class OptunaPruningCallback(tf.keras.callbacks.Callback):

    def __init__(self, trial: optuna.Trial, monitors: list[str],   
        debug: bool = True,
        print_every: int = 1,
        history_limit: int = 200,  # quanti trial passati usare per stats
        use_completed_only: bool = True,):
        super().__init__()

        self.print_every = print_every
        self.history_limit = history_limit
        self.use_completed_only = use_completed_only
        self.debug = debug 
        
        self.trial = trial

        self.regex: list[re.Pattern[str]] = []
        
        for monitor in monitors:
            self.regex.append(re.compile(rf"{monitor}"))
            

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # check if the metrics that we use for pruning is present in the logs in that epoch
        for r in self.regex:
            key = next((k for k in logs.keys() if r.search(k)), None)
            if key is not None: break

        if key is None: return

        # if it is present we use this value 
        value = float(logs[key])
        
        # if value if NaN (placeholder)
        if not math.isfinite(value): return

        # add to the report 
        self.trial.report(value, step=epoch)

        # Debug: stampa mediana e margine
        if self.debug and (epoch % self.print_every == 0):
            p25, med, p75, n = self._step_stats(epoch)

            if med is None:
                print(f"[Optuna] trial={self.trial.number} epoch={epoch} {key}={value:.6g} | no history yet")
            else:
                delta = value - med
                pct = (delta / med * 100.0) if med != 0 else float("inf")

                print(
                    f"[Optuna] trial={self.trial.number} epoch={epoch} {key}={value:.6g} "
                    f"| median={med:.6g} (n={n}, p25={p25:.6g}, p75={p75:.6g}) "
                    f"| margin={delta:+.6g} ({pct:+.2f}%)"
                )

        # Prune decision
        if self.trial.should_prune():
            # stampa anche motivo/margine al pruning
            if self.debug:
                p25, med, p75, n = self._step_stats(epoch)
                if med is None:
                    print(f"[Optuna] PRUNED trial={self.trial.number} epoch={epoch} ({key}={value:.6g})")
                else:
                    delta = value - med
                    pct = (delta / med * 100.0) if med != 0 else float("inf")
                    print(
                        f"[Optuna] PRUNED trial={self.trial.number} epoch={epoch} "
                        f"{key}={value:.6g} > median={med:.6g} "
                        f"by {delta:+.6g} ({pct:+.2f}%)"
                    )
            raise optuna.TrialPruned()
        


    def _step_stats(self, step: int) -> tuple[float | None, float | None, float | None, int]:
            vals = []
            for t in reversed(self.trial.study.trials):
                if self.use_completed_only and t.state != optuna.trial.TrialState.COMPLETE:
                    continue
                v = t.intermediate_values.get(step)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isfinite(fv):
                    vals.append(fv)
                if len(vals) >= self.history_limit:
                    break

            if not vals:
                return None, None, None, 0

            arr = np.asarray(vals, dtype=np.float64)
            p25 = float(np.percentile(arr, 25))
            med = float(np.percentile(arr, 50))
            p75 = float(np.percentile(arr, 75))
            return p25, med, p75, len(vals)