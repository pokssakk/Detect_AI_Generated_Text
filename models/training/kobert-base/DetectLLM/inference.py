# inference.py 는 모델 추론 및 submission 생성. 

import torch
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import AutoTokenizer

from dataset import ParagraphDataset
from utils import load_config, get_device, load_tokenizer, load_model

config = load_config()
device = get_device()
tokenizer = load_tokenizer(config['model']['name'])

test_dataset = ParagraphDataset(
    config['data']['test_path'], 
    tokenizer, 
    config['model']['max_length'], 
    config['model']['stride'], 
    is_test=True,
    cache_path=config['data']['test_cache_path']
    )
test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

model = load_model(config['model']['name'], config['model']['save_path'], device)

prob_dict = defaultdict(list)

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        para_ids = batch['paragraph_id']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs['logits']).squeeze(1).cpu().tolist()

        for pid, prob in zip(para_ids, probs):
            prob_dict[pid.item()].append(prob)

# 문단 단위 집계 (max pooling)
final_probs = {pid: max(probs) for pid, probs in prob_dict.items()}

# submission.csv 불러오기 & 값 대입
submission = pd.read_csv(config['output']['submission_path'], encoding='utf-8-sig')
submission['generated'] = submission.index.map(lambda i: final_probs.get(i, 0.0))
submission.to_csv(config['output']['submission_path'], index=False, encoding='utf-8-sig')

print(f"[✓] 최종 제출 파일 저장 완료 → {config['output']['submission_path']}")