from ..dataclasses.model_config import ModelConfig
from ..dataclasses.phase_config import MaskedModelingPhase
from ..dataclasses.training_config import TrainingConfig
from ..dataclasses.data_config import DataConfig

from ..enums.decoder_mode import DecoderMode
from ..enums.mask_mode import MaskMode
from ..enums.phase_name import PhaseName

def check_correctness(model_cfg: ModelConfig, training_cfg: TrainingConfig, data_cfg: DataConfig):

    if data_cfg.dataset.traj_fraction > 1 or data_cfg.dataset.traj_fraction <= 0:
        raise ValueError("Can't set the traj_fraction greater than 1 or less equal than 0")
    
    if data_cfg.windowing.input_seq_len + data_cfg.windowing.output_seq_len > data_cfg.dataset.time_steps:
        raise ValueError("The sum of output_seq_len and input_seq_len can't be greater than the total time steps")

    if not (0.0 <= data_cfg.split.val_ratio < 1.0 and 0.0 <= data_cfg.split.test_ratio < 1.0 and (data_cfg.split.val_ratio + data_cfg.split.test_ratio) < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0,1) and val_ratio + test_ratio < 1")

    if len(training_cfg.curriculum) != len(training_cfg.phases):
        raise ValueError(f"Can't set the length of curriculum different from the number of phases")

    for h in training_cfg.curriculum:
        if h > data_cfg.windowing.output_seq_len or h == 0 or h < -1:
            raise ValueError(f"Can't set as horizon of curriculum a number greater than output_seq_len ({data_cfg.windowing.output_seq_len}), equal to 0 or less than -1")
    
    for p in training_cfg.phases:
        if p.name == PhaseName.MASKED_MODELING:
            if p.mask_prob <=0 or p.mask_prob >=1:
                raise ValueError("Can't set as mask_prob a number less or equal than 0 or greater or equal than 1")
    
    fr_curve_probe = next(p for p in training_cfg.fr_eval.probes if p.name == "fr_curve")
    for h in fr_curve_probe.out_steps:
        if isinstance(h,int):
            if h > data_cfg.windowing.output_seq_len or h <= 0:
                raise ValueError(f"Can't set as horizon of fr_curve a number greater than output_seq_len ({data_cfg.windowing.output_seq_len}), equal to 0 or less than -1")
            
    if model_cfg.decoder_mode == DecoderMode.FULL_SEQ:
        for p in training_cfg.phases:
            if p.name in {PhaseName.FULL_AUTOREGRESSIVE, PhaseName.SCHEDULED_SAMPLING}:
                raise ValueError(
                    f"Incompatible config: decoder_mode={model_cfg.decoder_mode} "
                    f"cannot be used with phase '{p.name}'. "
                    f"Use DecoderMode.STEP_WISE (or remove that phase)."
                )
    for fr in training_cfg.fr_eval.probes:
        if fr.p_eval <= 0 or fr.p_eval > 1:
            raise ValueError("p_eval must be in (0,1].")


def validate_masked_modeling_phase(p: MaskedModelingPhase) -> None:
    if p.mask_mode == MaskMode.CONSTANT:
        if p.mask_value is None:
            raise ValueError("MaskedModelingPhase: mask_mode=CONSTANT need mask_value.")

    elif p.mask_mode == MaskMode.NOISE:
        if p.noise_sigma is None:
            raise ValueError("MaskedModelingPhase: mask_mode=NOISE need noise_sigma.")
        
