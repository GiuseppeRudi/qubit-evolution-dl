import math
import numpy as np
import optuna
import tensorflow as tf
import re


class OptunaPruningCallback(tf.keras.callbacks.Callback):

    def __init__(
        self,
        trial: optuna.Trial,
        monitors: list[str],   
        debug: bool = True,
        print_every: int = 1,
        history_limit: int = 200,  # quanti trial passati usare per stats
        use_completed_only: bool = True
    ):
        
        super().__init__()

        self.print_every = print_every
        self.history_limit = history_limit
        self.use_completed_only = use_completed_only
        self.debug = debug 
        
        self.trial = trial

        self.regex: list[re.Pattern[str]] = [re.compile(rf"{m}") for m in monitors]

    def _is_multi_objective(self) -> bool:
        dirs = getattr(self.trial.study, "directions", None)
        return bool(dirs) and len(dirs) > 1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # check if the metrics that we use for pruning is present in the logs in that epoch
        key = None
        for r in self.regex:
            key = next((k for k in logs.keys() if r.search(k)), None)
            if key is not None: break

        if key is None: return

        # if it is present we use this value 
        value = float(logs[key])
        
        # if value is NaN (placeholder)
        if not math.isfinite(value): return

        is_mo = self._is_multi_objective()

        # customised report: save the best so far
        # (leggero: salva solo l’ultimo valore e il best)
        best_key = f"best_{key}"
        prev_best = self.trial.user_attrs.get(best_key)

        # regola best: loss -> min, altri -> max (adatta se vuoi)
        improved = False
        if prev_best is None:
            improved = True
        else:
            if "loss" in key:
                improved = value < float(prev_best)
            else:
                improved = value > float(prev_best)

        if improved:
            self.trial.set_user_attr(best_key, float(value))
            self.trial.set_user_attr(f"{best_key}_epoch", int(epoch))

        # Debug print
        if self.debug and (epoch % self.print_every == 0):
            if is_mo:
                print(f"[Optuna-MO] trial={self.trial.number} epoch={epoch + 1} {key}={value:.6g}")
            else:
                p25, med, p75, n = self._step_stats(epoch)
                if med is None:
                    print(f"[Optuna] trial={self.trial.number} epoch={epoch + 1} {key}={value:.6g} | no history yet")
                else:
                    delta = value - med
                    pct = (delta / med * 100.0) if med != 0 else float("inf")
                    print(
                        f"[Optuna] trial={self.trial.number} epoch={epoch + 1} {key}={value:.6g} "
                        f"| median={med:.6g} (n={n}, p25={p25:.6g}, p75={p75:.6g}) "
                        f"| margin={delta:+.6g} ({pct:+.2f}%)"
                    )

        # pruning SOLO se single-objective
        if not is_mo:
            self.trial.report(value, step=epoch)
            if self.trial.should_prune():
                if self.debug:
                    print(f"[Optuna] PRUNED trial={self.trial.number} epoch={epoch + 1} ({key}={value:.6g})")
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
            
            return (
                float(np.percentile(arr, 25)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 75)),
                len(vals),
            )