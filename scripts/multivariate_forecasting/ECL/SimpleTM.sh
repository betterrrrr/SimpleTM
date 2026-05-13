export CUDA_VISIBLE_DEVICES=0
model_name=SimpleTM

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0


python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.006 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0


python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 360 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0


python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 360 \
  --pred_len 192 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.006 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0


python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0


python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.006 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.0


python3 - <<'PY'
from collections import defaultdict
from pathlib import Path
import re

project_root = Path('.')
source_file = project_root / 'result_long_term_forecast.txt'
result_file = project_root / 'result_ECL.txt'

pair_order = [
    (96, 96),
    (96, 192),
    (360, 96),
    (360, 192),
    (720, 96),
    (720, 192),
]
target_pairs = set(pair_order)
target_prefix = 'ECL_ECL_'
runs_per_setting = 3

def write_empty_table(path: Path):
    lines = ['seq_len\tpred_len\tmean MSE\tmean MAE']
    for seq_len, pred_len in pair_order:
        lines.append(f'{seq_len}\t{pred_len}\tNA\tNA')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

if not source_file.exists():
    write_empty_table(result_file)
    print(f'[summary] {source_file} not found, wrote empty summary to {result_file}')
    raise SystemExit(0)

entries = []
current_setting = None
metric_re = re.compile(r'mse:([0-9eE+\-.]+),\s*mae:([0-9eE+\-.]+)')

for raw in source_file.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line:
        continue

    metric_match = metric_re.search(line)
    if metric_match and current_setting is not None:
        entries.append((
            current_setting,
            float(metric_match.group(1)),
            float(metric_match.group(2)),
        ))
        current_setting = None
        continue

    if line.startswith(target_prefix):
        current_setting = line
    else:
        current_setting = None

recent_metrics = defaultdict(list)
for setting, mse, mae in reversed(entries):
    parts = setting.split('_')
    if len(parts) < 4:
        continue

    try:
        seq_len = int(parts[2])
        pred_len = int(parts[3])
    except ValueError:
        continue

    key = (seq_len, pred_len)
    if key not in target_pairs:
        continue

    if len(recent_metrics[key]) < runs_per_setting:
        recent_metrics[key].append((mse, mae))

    if all(len(recent_metrics[k]) >= runs_per_setting for k in target_pairs):
        break

summary_lines = ['seq_len\tpred_len\tmean MSE\tmean MAE']
for seq_len, pred_len in pair_order:
    values = recent_metrics.get((seq_len, pred_len), [])
    if not values:
        summary_lines.append(f'{seq_len}\t{pred_len}\tNA\tNA')
        continue

    mse_mean = sum(v[0] for v in values) / len(values)
    mae_mean = sum(v[1] for v in values) / len(values)
    summary_lines.append(f'{seq_len}\t{pred_len}\t{mse_mean:.4f}\t{mae_mean:.4f}')

result_file.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
print(f'[summary] wrote aggregated ECL results to {result_file}')
PY
