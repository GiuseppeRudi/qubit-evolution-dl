import optuna
import pandas as pd
from pathlib import Path

study_name = "hybrud_lstm_lvl1"
out_root = Path("runs/tuning") / study_name
storage = f"sqlite:///{(out_root / 'optuna.db').as_posix()}"

study = optuna.load_study(study_name=study_name, storage=storage)

df = study.trials_dataframe(("number","value","duration","params","user_attrs","system_attrs","state"))
df.to_csv(out_root / "report.csv", index=False)

print("Saved report in:", out_root / "report.csv")
print("Total trials:", len(study.trials))
