import keras
from src.qubit.core.data import prepare_dataset
from src.qubit.core.utils import get_device, parse_args
from src.qubit.core.config_loader import load_run_config, load_model_config, load_training_config, load_plot_config, load_data_config
from src.qubit.registry import get_builder,get_trainer
from src.qubit.core.save import save_outputs, make_run_output_dir,save_log
from src.qubit.core.error import check_correctness

import src.qubit.rnn.builders  
import src.qubit.transformer.builders
import src.qubit.training.rnn_trainer
import src.qubit.training.trn_trainer


def main():

    get_device()

    args = parse_args()
    run_cfg = load_run_config(args.run_cfg)
 
    data_cfg = load_data_config(run_cfg["data"])
    model_cfg = load_model_config(run_cfg["model"])
    training_cfg = load_training_config(run_cfg["training"])

    check_correctness(model_cfg,training_cfg,data_cfg)

    splits, feat_names = prepare_dataset(data_cfg)

    run_dir = make_run_output_dir(model_cfg)
    save_log(run_dir)
    
    builder = get_builder(model_cfg.type, model_cfg.variant,model_cfg.decoder_mode)
    model = builder(splits.X_train, splits.Y_train, model_cfg, args.model)

    TrainerCls = get_trainer(model_cfg.type)

    trainer = TrainerCls(model, model_cfg, training_cfg)
    history = None
    
    if args.training is True :
        print(f"\n--- Training: {model_cfg.name} [{model_cfg.type.value}/{model_cfg.variant.value}]  ---")
        history = trainer.fit(splits)

        print(f"Final loss: {history.history['loss'][-1]:.4f}")

    sample_x,sample_y, pred = trainer.predict_all_test(splits)
    trainer.report_sample(sample_x, sample_y, pred)
    

    plot_cfg = load_plot_config(run_cfg["plot"])

  
    save_outputs(splits, pred, model_cfg, feat_names, history, run_dir, training_cfg.fr_eval.split + "_fr_", plot_cfg, training_cfg, splits.Y_train.shape[1], model)  

if __name__ == "__main__":
    main()


# TODO scheduled sampling con noise injection 
# - name: scheduled_sampling_with_noise
#   epochs: 20
#   tf_ratio_start: 1.0
#   tf_ratio_end: 0.0
#   noise_injection:
#     enabled: true
#     noise_type: "adaptive"  # Cresce con la distanza dall'inizio
#     sigma_start: 0.0
#     sigma_end: 0.5  # ← Simula errori di magnitudine ~0.5


# TODO IMPORTANTE 
# 2) Problema strutturale con il curriculum: self._strategy è un oggetto Python dentro un train_step che Keras mette in graph
# Keras (con run_eagerly=False) wrappa train_step in tf.function. Dentro una tf.function:
# gli oggetti Python catturati (come self._strategy) diventano “costanti” del trace
# quando tu fai set_context(... strategy=...) e cambi strategia per fase/epoch, la graph potrebbe NON “vedere” il cambio oppure forzare retracing in modo brutto (memory/instabilità/errori strani)
# Tu hai già fatto bene a mettere ctx_epoch/ctx_total_epochs/ctx_horizon come tf.Variable per evitare retracing… ma la strategia resta Python, quindi il problema rimane.

