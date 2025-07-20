import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from transformers import AutoTokenizer, AutoModelForCausalLM
from konlpy.tag import Okt

okt = Okt()

# Load pretrained models
catboost_model = joblib.load("catboost_final_model.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
gpt2 = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2").to(device)
gpt2.eval()

# Feature extraction of input text
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[-1] == 0:
        return float('inf')
    with torch.no_grad():
        outputs = gpt2(**inputs, labels=input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()

# Prediction function
def predict_generated_prob(text):
    text = str(text)
    unique_ratio = get_unique_ratio(text)
    verb_ratio = get_verb_ratio(text)
    entropy = get_entropy(text)
    ppl = get_ppl(text)

    base_features = np.array([[unique_ratio, verb_ratio, entropy]])
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(base_features)

    feature_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['unique_ratio', 'verb_ratio', 'entropy']))
    feature_df['ppl'] = ppl

    prob = catboost_model.predict_proba(feature_df)[:, 1][0]
    return prob
