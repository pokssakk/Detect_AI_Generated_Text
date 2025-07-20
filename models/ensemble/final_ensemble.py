import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from konlpy.tag import Okt
from sklearn.preprocessing import PolynomialFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
okt = Okt()

# Load Fine-tuned Models
# 1. roberta-large
class RobertaClassifier(nn.Module):
    def __init__(self, model_name="klue/roberta-large", num_labels=1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

rob_model = RobertaClassifier()
state_dict = torch.load("a+r_roberta-large.pt", map_location="cpu")
rob_model.load_state_dict(state_dict)
rob_model.eval()
rob_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

# 2. roberta-base
class RobertaBaseClassifier(nn.Module):
    def __init__(self, model_name="klue/roberta-base", num_labels=1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

roberta_base_model = RobertaBaseClassifier()
roberta_base_state_dict = torch.load("a+r_roberta-base.pt", map_location="cpu")
roberta_base_model.load_state_dict(roberta_base_state_dict)
roberta_base_model.eval()
roberta_base_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

# 3. catboost (style&ppl based)
catboost_model = joblib.load("catboost_final_model.pkl")
gpt2_tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
gpt2_model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2").to(device)
gpt2_model.eval()




# Style & Perplexity (PPL) Feature Extraction for CatBoost
# style_feature : unique_ratio, verb_ratio, entropy의 polynomial features
# ppl : kogpt2
def get_unique_ratio(text):
    words = okt.morphs(text)
    return len(set(words)) / len(words) if words else 0.0

def get_verb_ratio(text):
    pos_tags = okt.pos(text)
    verb_count = sum(1 for _, tag in pos_tags if tag == 'Verb')
    return verb_count / len(pos_tags) if pos_tags else 0.0

def get_entropy(text):
    words = okt.morphs(text)
    word_freq = Counter(words)
    probs = np.array(list(word_freq.values())) / len(words) if words else np.array([1])
    return -np.sum(probs * np.log2(probs)) if words else 0.0

def get_ppl(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[-1] == 0:
        return float('inf')
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=input_ids).loss
        return torch.exp(loss).item()

def predict_catboost_prob(text):
    unique_ratio = get_unique_ratio(text)
    verb_ratio = get_verb_ratio(text)
    entropy = get_entropy(text)
    ppl = get_ppl(text)
    base_features = np.array([[unique_ratio, verb_ratio, entropy]])
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(base_features)
    feature_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['unique_ratio', 'verb_ratio', 'entropy']))
    feature_df['ppl'] = ppl
    return catboost_model.predict_proba(feature_df)[:, 1][0]



# Ensemble Strategy (Custom extreme voting)
'''
- If two or more models predict >=0.5 → take the maximum probability 
- Otherwise → take the minimum probability 
'''
def extreme_voting(probs):
    above = [p for p in probs if p >= 0.5]
    below = [p for p in probs if p < 0.5]
    if len(above) >= 2:
        final_prob = max(above)
    else:
        final_prob = min(below)
    final_label = int(final_prob >= 0.5)
    return final_prob, final_label



# Final Ensemble Prediction Function
def predict_ensemble_roberta_style(text):
    rob_inputs = rob_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        rob_large_prob = torch.sigmoid(rob_model(**rob_inputs)).item()

    rob_base_inputs = roberta_base_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        rob_base_prob = torch.sigmoid(roberta_base_model(**rob_base_inputs)).item()

    cat_prob = predict_catboost_prob(text)

    final_prob, final_label = extreme_voting([rob_large_prob, rob_base_prob, cat_prob])
    return {
        "roberta_large_prob": rob_large_prob,
        "roberta_base_prob": rob_base_prob,
        "styleppl_prob": cat_prob,
        "final_prob": final_prob,
        "final_label": final_label
    }
