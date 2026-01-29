from ..enums.decoder_mode import DecoderMode
import tensorflow as tf

from typing import cast
from ..dataclasses.phase_config import *

# Need to update the StategyLayer to track the different parameter for the custom loop
class PhaseSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(
            self,
            phases: list[PhaseConfig],
            curriculum: list[int],
            lr_global: float,
            clip_global: float,
            decoder_mode: DecoderMode,
            fr_eval=None
        ):
        super().__init__()
        self.phases = phases
        self.curriculum = curriculum
        self.lr_global = lr_global # learning rate 
        self.clip_global = clip_global
        self.decoder_mode = decoder_mode
        self.fr_eval = fr_eval # free running evaluations
        
        # list of indexes (starts of each epoch)
        self.bounds = [0]
        s = 0
        for ph in phases:
            s += ph.epochs
            self.bounds.append(s)


    def on_epoch_begin(self, epoch, logs=None):

        phase_idx, epoch_in_phase = self._phase_of(epoch)
        
        phase = self.phases[phase_idx]
        m = self.model

        # horizon for a specific phase 
        horizon = self.curriculum[phase_idx]

        # if in the file yaml want to use the global output_seq_len
        if horizon == -1: m.rt.horizon.assign(m.rt.t_out)
        else: m.rt.horizon.assign(horizon)

        m.rt.epoch_in_phase.assign(epoch_in_phase)
        m.rt.phase_epochs.assign(phase.epochs)

        # convert in  phase_id for the graph mode
        pid = {
            PhaseName.TEACHER_FORCING.value: 0,
            PhaseName.MASKED_MODELING.value: 1,
            PhaseName.SCHEDULED_SAMPLING.value: 2,
            PhaseName.FULL_AUTOREGRESSIVE.value: 3
        }[phase.name]
        
        m.rt.phase_id.assign(pid)
        
        # only if the current model is Hybrid so work with either STEP_WISE or the FULL_SEQ
        if self.decoder_mode == DecoderMode.HYBRID:
            if pid == 0 or pid == 1:
                m.rt.decoder_mode_id.assign(0)
            else: m.rt.decoder_mode_id.assign(1)

        # strategy-specific parameter

        if pid == 1:  # Masked Modeling 
            mask_mode_to_id = {MaskMode.ZERO.value: 0, MaskMode.CONSTANT.value: 1, MaskMode.NOISE.value: 2}[cast(MaskedModelingPhase, phase).mask_mode]
            mask_scope_to_id = {MaskScope.TIME.value: 0, MaskScope.FEATURE.value: 1, MaskScope.ELEMENT.value: 2}[cast(MaskedModelingPhase, phase).mask_scope]

            m.rt.mask_prob.assign(cast(MaskedModelingPhase, phase).mask_prob)
            m.rt.mask_mode_id.assign(mask_mode_to_id)
            m.rt.mask_scope_id.assign(mask_scope_to_id)
            m.rt.mask_value.assign(cast(MaskedModelingPhase, phase).mask_value)
            m.rt.noise_sigma.assign(cast(MaskedModelingPhase, phase).noise_sigma)
            m.rt.noise_replace.assign(cast(MaskedModelingPhase, phase).noise_replace)

        if pid == 2:  # Scheduled Sampling 
            ratio_mode_to_id = {RatioMode.LINEAR.value: 0, RatioMode.COSINE.value: 1, RatioMode.SIGMOID.value: 2, RatioMode.POWER.value: 3}[cast(ScheduledSamplingPhase, phase).ratio_mode]

            m.rt.tf_ratio_start.assign(cast(ScheduledSamplingPhase, phase).tf_ratio_start)
            m.rt.tf_ratio_end.assign(cast(ScheduledSamplingPhase, phase).tf_ratio_end)
            m.rt.per_feature.assign(cast(ScheduledSamplingPhase, phase).per_feature)
            m.rt.stop_grad_pred.assign(cast(ScheduledSamplingPhase, phase).stop_grad_pred)
            m.rt.ratio_mode.assign(ratio_mode_to_id)
            m.rt.mid_point.assign(cast(ScheduledSamplingPhase, phase).mid_point)
            m.rt.sharpness.assign(cast(ScheduledSamplingPhase, phase).sharpness)
            m.rt.power_value.assign(cast(ScheduledSamplingPhase, phase).power_value)
        
        if pid == 3:  # FullAutoregressive
            m.rt.gradient_through_time.assign(cast(FullAutoregressivePhase, phase).gradient_through_time)

        # if a specific phase don't have the private values of learning rate and clip_norm 
        # we set the global one
        lr = self.lr_global if phase.learning_rate is None else phase.learning_rate
        clip = self.clip_global if phase.clip_norm is None else phase.clip_norm
        if m.optimizer is not None: m.optimizer.learning_rate.assign(lr)
        m.current_clip_norm = clip

        # Info print at the start of each phase
        if epoch_in_phase == 0:
            actual_horizon = horizon if horizon != -1 else m.rt.t_out.numpy()
            print(f"\n{'='*70}")
            print(f"   PHASE {phase_idx+1}/{len(self.phases)}: {phase.name}")
            for key, value in phase.__dict__.items():
                if key not in ("epochs", "name", "learning_rate", "clip_norm"):
                    print(f"   {key.replace('_',' ').title()}: {value}")
            print(f"   Epochs: {phase.epochs}")
            print(f"   Learning Rate: {lr}")
            print(f"   Clipping Norm: {clip}")
            print(f"   Horizon (train loss): {actual_horizon}")
            print(f"   Output timesteps (val_loss and fr_loss): {m.rt.t_out.numpy()}")
            print(f"{'='*70}\n")

        # update callback free_running
        if self.fr_eval is not None:
            actual_horizon = horizon if horizon != -1 else m.rt.t_out.numpy()
            self.fr_eval.phase_horizon = actual_horizon
            self.fr_eval.end_of_phase = (epoch_in_phase == phase.epochs - 1)
            
    def _phase_of(self, epoch):
        # return the (phase_idx and interally epoch phase) of a current epoch
        for i in range(len(self.phases)):
            if self.bounds[i] <= epoch < self.bounds[i+1]:
                return i, epoch - self.bounds[i]
            
        # for the last phase 
        return len(self.phases)-1, self.phases[-1].epochs-1
