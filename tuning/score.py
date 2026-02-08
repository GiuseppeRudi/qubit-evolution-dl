from __future__ import annotations
from typing import Any
import math
import numpy as np
from qubit.utils.config_keys import FR_EVAL, OUTPUT_SEQ_LEN, FR_EVAL_SPLIT, OUT_STEPS, PROBE_NAME, PROBES, TRAINING, DATA,WINDOWING,CURRICULUM



def _pick_fr_target(metrics: dict, split: str, output_seq_len: int) -> float:
    fr_target_key = f"{split}_fr_target_loss_{output_seq_len}"

    vals = metrics.get(fr_target_key)
    last = vals[-1] if isinstance(vals, list) and vals else None
    if last is None or not math.isfinite(last):
        raise ValueError(f"The last value of {fr_target_key} cannot be None or infinite")

    return last

def _pick_fr_curve(metrics: dict[str, Any], *, split: str, curve_steps: list[int]) -> float:
    num = 0.0
    den = 0.0

    for s in curve_steps:
        key = f"{split}_fr_curve_loss_{s}"
        vals = metrics.get(key)
        last = vals[-1] if isinstance(vals, list) and vals else None

        if last is None or not math.isfinite(last):
            raise ValueError(f"The last value of {key} cannot be None or infinite")
        
        w = float(s)
        num += w * last
        den += w
    
    if den == 0.0:
        raise ValueError(f"The denominator of weighted sum for {split}_fr_curve_loss cannot be 0")

    return (num / den)


def _pick_fr_phase(metrics: dict, split: str, last_curriculum : int) -> float :
    fr_phase_key = f"{split}_fr_phase_loss_{last_curriculum}"

    vals = metrics.get(fr_phase_key)

    v = None

    if isinstance(vals, list) and vals:
        for i in range(len(vals) - 1, -1, -1):
            if not np.isnan(vals[i]):
                v = vals[i]
                break

    # print(f"fr_phase_loss = {v}")

    if v is None or not math.isfinite(v):
        raise ValueError(f"The value of {fr_phase_key} cannot be None or infinite")
        
    return v

def compute_score(metrics: dict[str, Any], base_cfg: dict, override: dict, level: int) -> float:

    #  TODO in the future this case can be modify from yaml confi for optuna
    
    # level 1 SCORE = 0.70 * fr_taget + 0.25 * fr_curve + 0.05 * fr_phase
    
    # level 2 SCORE = 0.80 * fr_target + 0.20 * val_loss
    # Since in this level we tune the ouput_seq_len we can't calculate the score 
    # using fr_curve and fr_phase because different trials can have different values and this is not FAIR
    # so in this case the formula score changes
    
    # level = 1 we take output_seq_len from the base_cfg yaml because optuna don't change it 
    # instead if we are in level = 2 we choose the ouput_seq_len from override
    
    if level == 1 :
        out_seq_len = base_cfg[DATA][WINDOWING][OUTPUT_SEQ_LEN]
    else :
        out_seq_len = override[DATA][WINDOWING][OUTPUT_SEQ_LEN] 

    # print(out_seq_len)
    
    split = base_cfg[TRAINING][FR_EVAL][FR_EVAL_SPLIT] 
    fr_target = _pick_fr_target(metrics, split, out_seq_len)


    if level == 2: 
        score = 0.80 * fr_target
        score += 0.20 * metrics["val_loss"][-1]

        return score 

    # other levels 
    curve_steps = []
    probes = base_cfg[TRAINING][FR_EVAL][PROBES]
    for p in probes:
        if p[PROBE_NAME] == "fr_curve":
            curve_steps = [round(x * out_seq_len) for x in p[OUT_STEPS]]
                                
    last_curriculum = int(base_cfg[TRAINING][CURRICULUM][-1] * out_seq_len)
    # print(last_curriculum)

    fr_curve  = _pick_fr_curve(metrics, split=split, curve_steps=curve_steps)
    fr_phase  = _pick_fr_phase(metrics, split=split,last_curriculum=last_curriculum)

    score = 0.70 * fr_target
    score += 0.25 * fr_curve
    score += 0.05 * fr_phase

    return score
