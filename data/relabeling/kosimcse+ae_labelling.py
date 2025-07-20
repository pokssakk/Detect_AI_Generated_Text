import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

# Basic settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "BM-K/KoSimCSE-roberta-multitask"



# Chunk generation using sliding window
def generate_chunk_csv(csv_path, tokenizer, max_len=128, stride=64, save_path="chunk_dataset.csv"):
    df = pd.read_csv(csv_path)
    df["paragraph_text"] = df["paragraph_text"].astype(str)
    df["combined_key"] = df["title"] + "||" + df["paragraph_text"]
    df["paragraph_id"] = df["combined_key"].factorize()[0]

    chunk_data = []
    for _, row in df.iterrows():
        tokens = tokenizer.encode(row["paragraph_text"], truncation=False, add_special_tokens=False)
        for start in range(0, len(tokens), stride):
            end = start + max_len
            chunk_tokens = tokens[start:end]
            chunk_decoded = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunk_data.append({
                "paragraph_id": row["paragraph_id"],
                "title": row["title"],
                "paragraph_text": row["paragraph_text"],
                "chunk_text": f"제목: {row['title']} 본문: {chunk_decoded}",
                "generated": row["generated"]
            })
            if end >= len(tokens):
                break
    pd.DataFrame(chunk_data).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[✓] {len(chunk_data)}개의 chunk 저장 완료 → {save_path}")



# Embedding & Cosine Similarity
def get_embeddings(texts, model_name, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    loader = DataLoader(texts, batch_size=batch_size)
    for batch in tqdm(loader, desc=f"임베딩 추출 중 ({model_name})"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0]
            embeddings.extend(emb.cpu().numpy())
    return np.array(embeddings)

def compute_cosine_similarity(embeddings):
    mean_vector = embeddings.mean(axis=0)
    return cosine_similarity(embeddings, [mean_vector]).flatten()



# AutoEncoder & Reconstruction Error
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def compute_ae_score(embeddings, epochs=30):
    scaler = StandardScaler()
    inputs_scaled = scaler.fit_transform(embeddings)
    ae = AutoEncoder(embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    inputs = torch.tensor(inputs_scaled, dtype=torch.float32).to(device)

    for _ in range(epochs):
        ae.train()
        optimizer.zero_grad()
        outputs = ae(inputs)
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()

    ae.eval()
    with torch.no_grad():
        errors = torch.mean((ae(inputs) - inputs) ** 2, dim=1).cpu().numpy()
    return errors



# Data preparation & Scoring
tokenizer = AutoTokenizer.from_pretrained(model_name)
generate_chunk_csv("train_sampled_10.csv", tokenizer, save_path="chunk_dataset.csv")

chunk_df = pd.read_csv("chunk_dataset.csv")
embeddings = get_embeddings(chunk_df["chunk_text"].tolist(), model_name)
chunk_df["cosine_similarity"] = compute_cosine_similarity(embeddings)
chunk_df["ae_score"] = compute_ae_score(embeddings)

# Threshold-based prediction
df_g0 = chunk_df[chunk_df["generated"] == 0]
cos_threshold = df_g0["cosine_similarity"].mean() - 2 * df_g0["cosine_similarity"].std()
ae_threshold = df_g0["ae_score"].mean() + 2 * df_g0["ae_score"].std()

chunk_df["chunk_predicted"] = (
    (chunk_df["cosine_similarity"] < cos_threshold) |
    (chunk_df["ae_score"] > ae_threshold)
).astype(int)



# Paragraph-level aggregation & Correction
paragraph_level = chunk_df.groupby("paragraph_id")["chunk_predicted"].max().reset_index()
meta = chunk_df[["paragraph_id", "title", "paragraph_text", "generated"]].drop_duplicates()
final_df = meta.merge(paragraph_level, on="paragraph_id", how="left")
final_df.rename(columns={"chunk_predicted": "paragraph_predicted"}, inplace=True)

# Correction: For generated=1 titles with no predicted positives
scores = chunk_df.groupby(["title", "paragraph_text"]).agg({"ae_score": "max", "cosine_similarity": "min"}).reset_index()
final_scored = final_df.merge(scores, on=["title", "paragraph_text"], how="left")

titles_to_fix = (
    final_df[final_df["generated"] == 1]
    .groupby("title")["paragraph_predicted"]
    .sum()[lambda x: x == 0]
    .index.tolist()
)

def choose_suspect(df):
    return df.sort_values(by=["ae_score", "cosine_similarity"], ascending=[False, True]).iloc[0].name

indices_to_update = [
    choose_suspect(final_scored[final_scored["title"] == title]) for title in titles_to_fix
]

final_scored["paragraph_predicted_corrected"] = final_scored["paragraph_predicted"]
final_scored.loc[indices_to_update, "paragraph_predicted_corrected"] = 1



# Final Result
final_summary = final_scored.rename(columns={"paragraph_predicted_corrected": "predicted"})
final_summary = final_summary[["title", "paragraph_text", "generated", "predicted"]]
final_summary.to_csv("train_paragraph_kosimcse_multitask.csv", index=False, encoding="utf-8-sig")
