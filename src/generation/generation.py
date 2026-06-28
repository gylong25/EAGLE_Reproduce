import os
import re
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams
import csv
from datetime import datetime


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='qwen7')
  parser.add_argument('--task', type=str, default='gsm8k')
  parser.add_argument('--dataset_split', type=str, default='test')
  parser.add_argument('--models_dir', type=str, default=os.getenv('MODELS_DIR', '/root/siton-data-zhangyajunData/gyl'))
  parser.add_argument('--output_dir', type=str,
                      default=os.getenv('OUTPUT_DIR', '/root/siton-data-zhangyajunData/gyl/EAGLE-main/output'))
  parser.add_argument('--temperature', type=float, default=0.0)
  parser.add_argument('--top_p', type=float, default=1.0)
  parser.add_argument('--max_tokens', type=int, default=1024)
  parser.add_argument('--batch_size', type=int, default=12)
  return parser.parse_args()


def setup_environment(args):
  with open('configs/model_config.json') as f:
    model_config = json.load(f)
  model_path_str = model_config['MODEL_STRUCTURE'][args.model]
  model_path = Path(model_path_str).expanduser().resolve()
  output_dir = Path(args.output_dir) / f"{args.task}_{args.dataset_split}_{args.model}"
  os.makedirs(output_dir, exist_ok=True)
  return model_path, output_dir


def initialize_llm(model_path):
  return LLM(
    model=os.fspath(model_path),
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    enforce_eager=True
  )


def create_sampling_params(args):
  stop_tokens = {
    'mmlu': ["\n"],
    'triviaqa': ["<|endoftext|>"],
    'gsm8k': ["endoftext"]
  }
  return SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    stop=stop_tokens[args.task],
    seed=1234
  )


def batch_generate_predictions(dataset, llm, sampling_params, prompt_fn, batch_size=8):
  predictions = []
  for i in range(0, len(dataset), batch_size):
    batch_indices = range(i, min(i + batch_size, len(dataset)))
    batch = [dataset[k] for k in batch_indices]
    prompts = [prompt_fn(item) for item in batch]
    outputs = llm.generate(prompts, sampling_params)

    for j, output in enumerate(outputs):
      pred = {
        "question": batch[j].get('question', batch[j]),
        "prediction": output.outputs[0].text.strip(),
        "reference": batch[j].get('answer', None)
      }
      predictions.append(pred)
  return predictions


def evaluate_predictions(predictions, dataset, task):
  if task == 'mmlu':
    correct = 0
    results = []
    for pred, ref in zip(predictions, dataset):
      pred_letter = re.match(r'[A-D]', pred['prediction'].upper())
      pred_letter = pred_letter.group(0) if pred_letter else "INVALID"
      ref_letter = ["A", "B", "C", "D"][ref['answer']]
      is_correct = pred_letter == ref_letter
      correct += is_correct
      results.append({**pred, "correct": is_correct})

  elif task == 'triviaqa':
    def normalize_text(text):
      if not text: return ""
      normalized = text.lower()
      normalized = re.sub(r"[^\w\s']", " ", normalized)
      return re.sub(r"\s+", " ", normalized).strip()

    correct = 0
    results = []
    for pred in predictions:
      pred_norm = normalize_text(pred['prediction'])
      true_answers = set()
      for field in ['aliases', 'normalized_aliases', 'value', 'normalized_value']:
        value = pred['reference'].get(field, [])
        items = value if isinstance(value, list) else [value]
        for item in items:
          norm_item = normalize_text(str(item))
          if norm_item: true_answers.add(norm_item)

      is_correct = any(
        re.search(r'(^|\s)' + re.escape(a) + r'($|\s)', pred_norm)
        for a in true_answers if a
      )
      correct += is_correct
      results.append({**pred, "correct": is_correct})

  elif task == 'gsm8k':
    def extract_number(text):
      match = re.findall(r'[-+]?\d[\d,]*\.?\d*', text)
      return match[-1].replace(',', '') if match else None

    correct = 0
    results = []
    for pred, ref in zip(predictions, dataset):
      pred_num = extract_number(pred['prediction'])
      ref_num = extract_number(ref['answer'])
      is_correct = pred_num and ref_num and float(pred_num) == float(ref_num)
      correct += is_correct
      results.append({**pred, "correct": is_correct})

  accuracy = correct / len(predictions)
  return accuracy, results


def save_results(output_dir, task, predictions, accuracy, detailed_results, args):
  pred_file = os.path.join(output_dir, f'{task}_predictions.json')
  with open(pred_file, 'w') as f:
    json.dump(predictions, f, indent=2)

  eval_file = os.path.join(output_dir, f'{task}_evaluation.json')
  with open(eval_file, 'w') as f:
    json.dump({"accuracy": accuracy, "results": detailed_results}, f, indent=2)

  csv_path = os.path.join(args.output_dir, 'evaluation_summary.csv')

  with open('configs/dataset_config.json') as f:
    dataset_config = json.load(f)
  prompt_template = dataset_config['DATASET_CONFIG'][args.task]['prompt_template'].replace('\n', '\\n')

  with open(csv_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if csvfile.tell() == 0:
      writer.writerow(["timestamp", "model_name", "task", "prompt_template",
                       "temperature", "top_p", "max_tokens", "accuracy"])
    writer.writerow([
      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      args.model, args.task, prompt_template,
      args.temperature, args.top_p, args.max_tokens, f"{accuracy:.4f}"
    ])


def main():
  args = parse_args()
  model_path, output_dir = setup_environment(args)

  with open('configs/dataset_config.json') as f:
    dataset_config = json.load(f)
  config = dataset_config['DATASET_CONFIG'][args.task]

  split_key = args.dataset_split
  file_path = config["data_files"][split_key]
  dataset = load_dataset("parquet", data_files={split_key: file_path})[split_key]  # 本地加载数据集

  llm = initialize_llm(model_path)
  sampling_params = create_sampling_params(args)

  if args.task == 'mmlu':
    def prompt_fn(ex):
      choices = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ex['choices']))
      return config['prompt_template'].format(question=ex['question'], choices=choices)
  else:
    prompt_fn = lambda ex: config['prompt_template'].format(question=ex['question'])
  predictions = batch_generate_predictions(dataset, llm, sampling_params, prompt_fn, args.batch_size)

  accuracy, detailed_results = evaluate_predictions(predictions, dataset, args.task)

  save_results(output_dir, args.task, predictions, accuracy, detailed_results, args)

  print(f"{args.task.upper()} Accuracy: {accuracy:.2%}")
  print(f"Correct: {int(accuracy * len(detailed_results))}/{len(detailed_results)}")


if __name__ == "__main__":
  main()