# dataset.py 는 전처리된 데이터는 PyTorch Dataset 형태로 정의하고, 토크나이저 적용 및 텐서 변환 등을 수행해서 입력 포맷화.

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle

class ParagraphDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len, stride, is_test=False, cache_path="samples.pkl"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.is_test=is_test

        if os.path.exists(cache_path):
            print(f"[✓] 캐시된 슬라이딩 샘플 로딩 중: {cache_path}")
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)

        else:
            print(f"[•] 슬라이딩 윈도우 적용 중... (처음 1회만 진행)")
            self.data = pd.read_csv(csv_path)
            self.samples = []

            for idx, row in self.data.iterrows():
                title = str(row['title'])
                paragraph = str(row['paragraph_text'])
                label = None if self.is_test else float(row['generated'])

                # 토큰 단위로 paragraph만 먼저 인코딩
                paragraph_tokens = tokenizer.encode(paragraph, truncation=False, add_special_tokens=False)

                # 슬라이딩 윈도우 적용
                for start in range(0, len(paragraph_tokens), stride):
                    end = start + max_len
                    chunk_tokens = paragraph_tokens[start:end]
                    if len(chunk_tokens) < 10:
                        break  # 너무 짧은 chunk는 버림
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    full_text = f"제목: {title} 본문: {chunk_text}"
                    if self.is_test:
                        self.samples.append((full_text, label, idx))
                    else:
                        self.samples.append((full_text, label))
                    if end >= len(paragraph_tokens):
                        break
            with open(cache_path, "wb") as f:
              pickle.dump(self.samples, f)

    def __len__(self): # 전체 문단 개수 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        if self.is_test:
            text, label, para_id = self.samples[idx]
        else:
            text, label = self.samples[idx]

        encoding = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True, # 문장이 max_length보다 길면 자동으로 잘라줌 -> 데이터 손실 가능. overflowing_token 변수 써도 되는데 그럼 모델 학습에서 복잡도 올라가서 일단 보류.
            max_length=self.max_len, 
            return_tensors='pt'
        )

        item = {
            'input_ids' : encoding['input_ids'].squeeze(0),
            'attention_mask' : encoding['attention_mask'].squeeze(0)
        }

        if self.is_test:
          item['paragraph_id']=para_id
        else:
          item['label']=torch.tensor(label, dtype=torch.float)
        return item
    

