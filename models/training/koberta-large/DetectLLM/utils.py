# 공통 함수 정의


import yaml
import torch
from transformers import AutoTokenizer
from model import RobertaClassifier

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def load_model(model_name, model_path, device):
    model = RobertaClassifier(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model