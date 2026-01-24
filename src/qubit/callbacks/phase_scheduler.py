import tensorflow as tf

from typing import cast
from ..dataclasses.phase_config import *
from ..enums.prediction_mode import PredictionMode

class PhaseSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self, phases: list[PhaseConfig], curriculum : list[int], lr_global : float, clip_global : float , fr_eval=None ):
        super().__init__()
        self.phases = phases
        self.curriculum = curriculum
        self.lr_global = lr_global
        self.clip_global = clip_global
        self.fr_eval = fr_eval
        
        # prefissi cumulativi delle epoche: [0, e0, e0+e1, ...]
        self.bounds = [0]
        s = 0
        for ph in phases:
            s += ph.epochs
            self.bounds.append(s)

    def _phase_of(self, epoch):
        # ritorna (phase_idx, epoch_in_phase)
        for i in range(len(self.phases)):
            if self.bounds[i] <= epoch < self.bounds[i+1]:
                return i, epoch - self.bounds[i]
        return len(self.phases)-1, self.phases[-1].epochs-1


    # Codice corretto completo:
    def on_epoch_begin(self, epoch, logs=None):
        phase_idx, e_in = self._phase_of(epoch)
        phase = self.phases[phase_idx]
        m = self.model

        # Determina horizon
        horizon_py = self.curriculum[phase_idx]
        if horizon_py == -1: m.rt.horizon.assign(m.rt.t_out)
        else: m.rt.horizon.assign(horizon_py)

        # Aggiorna contesto TF
        m.rt.epoch_in_phase.assign(e_in)
        m.rt.phase_epochs.assign(phase.epochs)
        # NON riassegnare horizon qui!

        # phase_id
        pid = {
            PhaseName.TEACHER_FORCING.value: 0,
            PhaseName.MASKED_MODELING.value: 1,
            PhaseName.SCHEDULED_SAMPLING.value: 2,
            PhaseName.FULL_AUTOREGRESSIVE.value: 3
        }[phase.name]
        m.rt.phase_id.assign(pid)

        # parametri strategy-specific
        if pid == 1:  # MM
            mask_mode_to_id = {MaskMode.ZERO.value: 0, MaskMode.CONSTANT.value: 1, MaskMode.NOISE.value: 2}[cast(MaskedModelingPhase, phase).mask_mode]
            mask_scope_to_id = {MaskScope.TIME.value: 0, MaskScope.FEATURE.value: 1, MaskScope.ELEMENT.value: 2}[cast(MaskedModelingPhase, phase).mask_scope]

            m.rt.mask_prob.assign(cast(MaskedModelingPhase, phase).mask_prob)
            m.rt.mask_mode_id.assign(mask_mode_to_id)
            m.rt.mask_scope_id.assign(mask_scope_to_id)
            m.rt.mask_value.assign(cast(MaskedModelingPhase, phase).mask_value)
            m.rt.noise_sigma.assign(cast(MaskedModelingPhase, phase).noise_sigma)
            m.rt.noise_replace.assign(cast(MaskedModelingPhase, phase).noise_replace)

        if pid == 2:  # SS
            m.rt.tf_ratio_start.assign(cast(ScheduledSamplingPhase, phase).tf_ratio_start)
            m.rt.tf_ratio_end.assign(cast(ScheduledSamplingPhase, phase).tf_ratio_end)
            m.rt.per_feature.assign(cast(ScheduledSamplingPhase, phase).per_feature)
        
        if pid == 3:  # FA
            m.rt.gradient_through_time.assign(cast(FullAutoregressivePhase, phase).gradient_through_time)

        # lr / clip
        lr = self.lr_global if phase.learning_rate is None else phase.learning_rate
        clip = self.clip_global if phase.clip_norm is None else phase.clip_norm
        if m.optimizer is not None:
            m.optimizer.learning_rate.assign(lr)
        m.current_clip_norm = clip

        # Print info solo all'inizio della fase
        if e_in == 0:
            actual_horizon = horizon_py if horizon_py != -1 else m.rt.t_out.numpy()
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

        # aggiorna callback FR
        if self.fr_eval is not None:
            actual_horizon = horizon_py if horizon_py != -1 else m.rt.t_out.numpy()
            self.fr_eval.phase_horizon = actual_horizon
            self.fr_eval.end_of_phase = (e_in == phase.epochs - 1)
            
