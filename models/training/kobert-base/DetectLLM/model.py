# model.py 에서는 모델 클래스 정의. 

import torch
import torch.nn as nn
from transformers import AutoModel

class RobertaClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float=0.1):
        super(RobertaClassifier, self).__init__() # 부모 클래스 호출
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(
            input_ids = input_ids, 
            attention_mask = attention_mask
        )

        # [CLS] 토큰 벡터만 추출하면 됨. 
        cls_output = outputs.last_hidden_state[:, 0, :]

        # fine-tuning 시작점
        dropped = self.dropout(cls_output)
        logits = self.classifier(dropped) # shape: (batch_size, 1)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss 

        return result

    