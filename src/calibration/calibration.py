import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import json
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from safetensors.torch import load_file

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='qwen7')
  parser.add_argument('--task', type=str, default='gsm8k')
  parser.add_argument('--dataset_split', type=str, default='test')
  parser.add_argument('--models_dir', type=str, default=os.getenv('MODELS_DIR', '/root/siton-data-zhangyajunData/gyl'))
  parser.add_argument('--output_dir', type=str,
                      default=os.getenv('OUTPUT_DIR', '/root/siton-data-zhangyajunData/gyl/EAGLE-main/output'))
  args = parser.parse_args()
  return args

def extract_lm_head_weight(model_path: Path, device: str = None) -> torch.Tensor:
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  lm_head_key = "lm_head.weight"
  safetensor_files = sorted(model_path.glob("*.safetensors"))

  if not safetensor_files:
    raise FileNotFoundError(f"No .safetensors files found in {model_path}")

  for file in safetensor_files:
    state_dict = load_file(file, device="cpu")
    if lm_head_key in state_dict:
      weight = state_dict[lm_head_key].float().to(device)
      print(f"Found `{lm_head_key}` in {file.name} with shape: {weight.shape}")
      return weight

  raise KeyError(f"{lm_head_key} not found in any of the .safetensors files in {model_path}")


def process_hidden_states(hidden_states_list, lm_head, tokenizer, apply_softmax=True):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  DIGIT_TOKEN_IDS = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]

  digits_outputs = []
  none_indices = []

  for idx, hs in enumerate(hidden_states_list):
    if hs is None:
      none_indices.append(idx)
      continue

    with torch.no_grad():
      hs_tensor = torch.tensor(hs, device=device).squeeze()
      hs_tensor_dtype = hs_tensor.dtype

      if lm_head.dtype != hs_tensor_dtype:
        lm_head_converted = lm_head.to(dtype=hs_tensor_dtype)
      else:
        lm_head_converted = lm_head

      logits = hs_tensor @ lm_head_converted.T
      digit_values = logits[:, DIGIT_TOKEN_IDS]

      if apply_softmax:
        digit_values = torch.softmax(digit_values, dim=-1)

      digits_outputs.append(digit_values.cpu())

  return digits_outputs, none_indices


def calibration(probs, y_true, n_bins=10):
  if torch.is_tensor(probs):
    probs = probs.detach().cpu().numpy()
  if torch.is_tensor(y_true):
    y_true = y_true.detach().cpu().numpy()

  bin_boundaries = np.linspace(0, 1, n_bins + 1)
  bin_lowers = bin_boundaries[:-1]
  bin_uppers = bin_boundaries[1:]

  confidences = probs
  accuracies = np.array(y_true)
  ece = 0.0

  for i in range(n_bins):
    in_bin = np.logical_and(confidences > bin_lowers[i], confidences <= bin_uppers[i])
    bin_size = np.sum(in_bin)

    if bin_size > 0:
      bin_acc = np.mean(accuracies[in_bin])
      bin_conf = np.mean(confidences[in_bin])
      ece += np.abs(bin_acc - bin_conf) * bin_size / len(confidences)

  return {
    'AUROC': roc_auc_score(y_true, probs),
    'AUPRC': average_precision_score(y_true, probs),
    'ECE_10': ece,
    'Correlation': np.corrcoef(probs, y_true)[0, 1],
    'Brier_Score': brier_score_loss(y_true, probs)
  }


def normalize(scores, method='9'):
  scores = (scores - 0) / (9 - 0)
  return np.clip(scores, 0, 1)


def aggregate(digitsLogits, layers=slice(20, None, None)):
  aggregated_logits = digitsLogits[layers, :, :].mean(dim=0)
  probs = F.softmax(aggregated_logits, dim=-1)
  score_range = torch.arange(digitsLogits.shape[-1], device=digitsLogits.device).float()
  scores = torch.sum(probs * score_range, dim=-1)
  return scores


def load_configs():
  with open('configs/model_config.json') as f:
    model_config = json.load(f)
  return model_config['MODEL_STRUCTURE'], model_config['EAGLE_CONFIG']


def main():
  MODEL_STRUCTURE, EAGLE_CONFIG = load_configs()
  args = parse_args()
  
  model_identifier = MODEL_STRUCTURE.get(args.model)
  if model_identifier is None:
    raise ValueError(f"Unknown model: {args.model}. Available models: {list(MODEL_STRUCTURE.keys())}")

  with open('configs/model_config.json') as f:
    model_config = json.load(f)
  model_path_str = model_config['MODEL_STRUCTURE'][args.model]
  model_path = Path(model_path_str).expanduser().resolve()
  dir_path = os.path.join(args.output_dir, f'{args.task}_{args.dataset_split}_{args.model}')
  dir_path = Path(args.output_dir) / f"{args.task}_{args.dataset_split}_{args.model}"

  if not (dir_path / 'digitsLogits_digitsSoftmax_noneId.npy').exists():
    data = np.load(dir_path / 'self_eval.npy', allow_pickle=True)
    self_eval_scores = np.array([entry['self_eval_scores'] for entry in data]).squeeze()
    hidden_states_list = [entry['hidden_states'] for entry in data]

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    lm_head_weight = extract_lm_head_weight(model_path)

    digits_softmax, none_softmax = process_hidden_states(hidden_states_list, lm_head_weight, tokenizer,
                                                         apply_softmax=True)
    digits_logits, none_logits = process_hidden_states(hidden_states_list, lm_head_weight, tokenizer,
                                                       apply_softmax=False)

    assert none_softmax == none_logits

    np.save(dir_path / 'digitsLogits_digitsSoftmax_noneId.npy', {
      'digitsLogits': digits_logits,
      'digitsSoftmax': digits_softmax,
      'id_nonetype': none_softmax
    })

  with open(dir_path / f'{args.task}_evaluation.json') as f:
    predictions = json.load(f)
  correctness = np.array([p['correct'] for p in predictions['results']]).astype(int)

  digits = np.load(dir_path / 'digitsLogits_digitsSoftmax_noneId.npy', allow_pickle=True).item()
  digitsLogits = torch.stack(digits['digitsLogits']).transpose(0, 1)
  id_nonetype = digits['id_nonetype']

  last_n = EAGLE_CONFIG.get(args.model, {}).get(args.task, 20)
  current_slice = slice(-last_n, None)

  metrics = ['AUROC', 'AUPRC', 'ECE_10', 'Correlation', 'Brier_Score']
  all_results = {metric: [] for metric in metrics}

  probs = normalize(aggregate(digitsLogits, current_slice), method='9')
  probs = np.clip(probs, 0, 1)

  y_true_temp = [x for idx, x in enumerate(correctness) if idx not in id_nonetype]
  y_true = y_true_temp[:len(probs)]

  if len(y_true) > 0 and len(probs) > 0:
    calibration_results = calibration(probs, y_true, n_bins=10)
    for metric in metrics:
      all_results[metric].append(calibration_results[metric])
  else:
    for metric in metrics:
      all_results[metric].append(np.nan)

  results = {
    'metrics': all_results,
    'config': {
      'model': args.model,
      'task': args.task,
      'last_n_layers': last_n,
      'slice_used': str(current_slice)
    }
  }

  output_file = dir_path / f'calibration_results_last{last_n}.json'
  with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  main()