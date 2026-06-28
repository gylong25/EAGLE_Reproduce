import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
import random


def set_seed(seed=1234):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


set_seed(1234)

with open('configs/model_config.json') as f:
  model_config = json.load(f)
with open('configs/dataset_config.json') as f:
  dataset_config = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='qwen7')
parser.add_argument('--task', type=str, default='gsm8k')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--top_p', type=float, default=1)
parser.add_argument('--max_tokens', type=int, default=1024)
parser.add_argument('--dataset_split', type=str, default='test')
parser.add_argument('--model_dir', type=str, default='/root/siton-data-zhangyajunData/gyl')
parser.add_argument('--output_dir', type=str, default='/root/siton-data-zhangyajunData/gyl/EAGLE-main/output')
parser.add_argument('--batch_size', type=int, default=12, help='批处理大小')
args = parser.parse_args()

MODEL_PATHS = {
  'qwen7': os.path.join(args.model_dir,
                        f"models--{model_config['MODEL_STRUCTURE']['qwen7'].replace('/', '--')}/snapshots/1")
}

model = args.model
task = args.task
temperature = args.temperature
top_p = args.top_p
max_tokens = args.max_tokens

with open('configs/model_config.json') as f:
  model_config = json.load(f)
model_path_str = model_config['MODEL_STRUCTURE'][args.model]
model_path = Path(model_path_str).expanduser().resolve()
dir_path = os.path.join(args.output_dir, f'{task}_{args.dataset_split}_{args.model}')

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)

with open(os.path.join(dir_path, f'{task}_evaluation.json')) as f:
  predictions = json.load(f)
questions = [p['question'] for p in predictions['results']]
answers = [p['prediction'] for p in predictions['results']]


class SelfEvaluator:
  def __init__(self, model, tokenizer):
    self.tokenizer = tokenizer
    self.model = model
    self.DIGIT_TOKEN_IDS = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]

  def tokenize(self, texts):
    enc = self.tokenizer(
      texts,
      return_tensors="pt",
      padding=True,
      truncation=True
    ).to(self.model.device)
    return enc

  def find_first_digit_position(self, sequences, input_length):
    for i, token_id in enumerate(sequences[0][input_length:]):
      if int(token_id) in self.DIGIT_TOKEN_IDS:
        return i
    return None

  def evaluate(self, batch_qs, batch_as):
    prompts = []
    for q, a in zip(batch_qs, batch_as):
      prompt = (
        f"Question:\n{q}\n\n"
        f"Answer:\n{a}\n\n"
        "Rate how likely the answer is to be correct using a number from 0 to 9:\n"
        "- 0 = Very likely incorrect\n"
        "- 5 = Uncertain or partially correct\n"
        "- 9 = Very likely correct\n"
        "Confidence score (0-9): "
      )
      prompts.append(prompt)

    batch = self.tokenize(prompts)
    with torch.no_grad():
      outputs = self.model.generate(
        **batch,
        do_sample=True,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        output_hidden_states=True,
        output_logits=True,
        return_dict_in_generate=True
      )

    input_length = batch['input_ids'].shape[1]
    batch_size = len(batch_qs)
    batch_results = []

    for b in range(batch_size):
      gen_tokens = outputs.sequences[b][input_length:]
      pos = None
      for i, token_id in enumerate(gen_tokens):
        if int(token_id) in self.DIGIT_TOKEN_IDS:
          pos = i
          break

      hidden_states = None
      if pos is not None:
        hidden_states = [
          [token_hidden[b, -1, :].cpu().numpy()]
          for token_hidden in outputs.hidden_states[pos]
        ]

      digit_probs = []
      if pos is not None:
        logits = outputs.logits[pos][b].cpu()
        probs = torch.softmax(logits, dim=-1)
        digit_probs = [
          probs[id].item() if id < probs.shape[-1] else 0.0
          for id in sorted(self.DIGIT_TOKEN_IDS)
        ]
      else:
        digit_probs = [0.0] * 10

      generated_texts = self.tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True
      )
      batch_results.append({
        "self_eval_scores": np.array([digit_probs]),
        "hidden_states": hidden_states,
        "generated_texts": generated_texts[0],
        "self_eval_scores_all": outputs.logits[pos][b].cpu() if pos is not None else None,
      })

    return batch_results


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
evaluator = SelfEvaluator(model, tokenizer)

# results = []
# total = len(questions)
# batch_size = args.batch_size
# print(f"[自评估] 即将并行评估 {total} 条样本，Batch Size = {batch_size}...")
# for i in range(0, total, batch_size):
#     batch_qs = questions[i: i + batch_size]
#     batch_as = answers[i: i + batch_size]

#     batch_results = evaluator.evaluate(batch_qs, batch_as)
#     results.extend(batch_results)

#     # 适当降低打印频率，避免频繁 I/O 影响速度
#     if (i // batch_size) % 5 == 0 or (i + batch_size) >= total:
#         print(f"[自评估] 进度：{min(i + batch_size, total)}/{total}")

# print("[自评估] 完成。")
# np.save(os.path.join(dir_path, 'self_eval.npy'), results)

# ==================== 断点续跑改动开始 ====================
out_file = os.path.join(dir_path, 'self_eval.npy')

# 1. 尝试读取已有的断点数据
if os.path.exists(out_file):
  try:
    results = np.load(out_file, allow_pickle=True).tolist()
    start_idx = len(results)
    print(f"[自评估] 检测到断点文件，已自动加载 {start_idx} 条结果，将从第 {start_idx} 条继续运行...")
  except Exception as e:
    print(f"[自评估] 读取断点文件失败 ({e})，将从头开始运行...")
    results = []
    start_idx = 0
else:
  results = []
  start_idx = 0

total = len(questions)
batch_size = args.batch_size
print(f"[自评估] 即将评估剩余的 {total - start_idx}/{total} 条样本，Batch Size = {batch_size}...")

# 2. 将 range 的起点改为 start_idx
for i in range(start_idx, total, batch_size):
  batch_qs = questions[i: i + batch_size]
  batch_as = answers[i: i + batch_size]

  batch_results = evaluator.evaluate(batch_qs, batch_as)
  results.extend(batch_results)

  # 3. 每跑完 5 个 Batch（或到最后），就实时覆盖保存一次，防止意外中断
  if (i // batch_size) % 5 == 0 or (i + batch_size) >= total:
    print(f"[自评估] 进度：{min(i + batch_size, total)}/{total}")
    np.save(out_file, results)  # 实时写入硬盘

print("[自评估] 全部完成。")
# ==================== 断点续跑改动结束 ====================