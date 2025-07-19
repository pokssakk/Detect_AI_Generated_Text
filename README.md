# Detect LLM: Paragraph-level Classification of AI-Generated Text
<img width="1749" height="1256" alt="image" src="https://github.com/user-attachments/assets/53d6584f-be0f-4f14-8584-37c6678d754d" />


## :round_pushpin: Overview
This repository presents our solution for the **Detect AI-generated Text Competition**, a national-level data science competition hosted in 2025, Dacon.

The goal of the cahllenge was to detect whether a **paragraph** was written by a Large Language Model (LLM), with only document-level labels provided. This required re-egineering the data, designing weak supervision strategies, and buildig robust classifiers in a highly imbalanced setting.

[Competition Link](https://dacon.io/competitions/official/236473/overview/description)

<br>

## :round_pushpin: Competition Information

- **Host**: Dacon
- **Track**: Detect AI-generated Text
- **Evaluation Metric**: ROC-AUC
- **Input**: Document-level 'full_text' (Provided as train.csv)
- **Label**: 'generated' (0 for human-written, 1 for AI-generated)
- **Goal**: Classify each paragraph as human- or AI-generated

### ✔️ train.csv
|title|full_text|generated|
|------|---|---|
|카호올라웨섬|카호올라웨섬은 하와이 제도를 구성하는 (중략...) |0|
|청색거성|천문학에서 청색거성은 광도 분류 (중략...) |0|
|수난곡|수난곡은 배우의 연기 없이 무대에 (중략...) |1|

### ✔️ test.csv
|ID|title|paragraph_index|paragraph_text|
|---|------|---|---|
|TEST_0000|공중 도덕의 의의와 필요성|0|도덕이란 원래 개인의 자각...|
|TEST_0001|공중 도덕의 의의와 필요성|1|도덕은 단순히 개인의 문제...|
|TEST_0002|공중 도덕의 의의와 필요성|2|여기에 이른바 공중도덕은...|

### ✔️ sample_submission.csv
|ID|generated|
|---|------|
|TEST_0000|0|
|TEST_0001|0|

<br>

## :round_pushpin: Data Analysis & Preprocessing

Our core challenge was to reconstruct paragraph-level labels from document-level data in an extremely imbalanced and noisy setting. To overcome this, we developed multiple **weak supervision and filtering strategies**, focusing on data-centric approaches.

<br>
<p align='center'>
<img width="699" height="268" alt="image" src="https://github.com/user-attachments/assets/0048a66a-171c-4ca7-a485-4e4a78ac55ce" />

### ➡️ 1) Data Restructuring
- Each `full_text` (up to 9 paragraphs) was split into multiple `paragraph_text` units
- For long paragraphs, we applied **sliding window chunking** with overlapping stride
- Input format:  
  `"제목: {title} 본문: {paragraph_text}"`

### ➡️ 2) Class Imbalance Handling
The dataset exhibited **severe class imbalance**:  
approximately 10:1 ratio between `generated=0` and `generated=1` labels.

To address this, we performed both **data augmentation** and **filtering** for high-confidence positive samples.
- Oversampling + Label Propagation
- Filtering using Perplexity and Semantic Similarity

### 🔁 3) Data Augmentation (Generated = 1)

We used a pretrained KoGPT model ("kanana") to generate synthetic paragraphs mimicking `generated=1` style.  
These augmented paragraphs were added to the training set to enrich the positive class.

- Model: kanana LLM (fine-tuned KoGPT2)
- Prompt-based generation conditioned on style
- Manual filtering + confidence scoring

### 🔁 4) Weak Labeling & Filtering Strategies  [learn more->](https://github.com/pokssakk/data-experiments)

We experimented with various **heuristic and unsupervised labeling techniques**:

#### 1. Perplexity-based Filtering
- Used GPT-like models to compute **perplexity scores**
- Sentences or paragraphs with low perplexity were considered likely AI-generated
- Applied thresholding to extract `generated=1` candidates

#### 2. Style Feature + PPL Clustering
- Extracted syntactic style features (e.g., average sentence length, token diversity)
- Combined with perplexity
- Performed **KMeans or HDBSCAN** clustering to isolate AI-like patterns

#### 3. HDBSCAN on Paragraph-level PPL
- Applied sentence-wise perplexity estimation
- Clustered using **HDBSCAN** to extract dense positive clusters
- Label assigned to entire paragraph if one or more sentences clustered as AI

#### 4. Perturbation-based Confidence Scoring
- Modified (perturbed) the sentence input
- Measured change in log-likelihood (LL) or model confidence
- Used as a proxy for model “surprise” or fluency robustness


<br>

## :round_pushpin: Modeling Strategy

### ➡️ Base Models:
The first baseline model used KLUE-RoBERTa (Which you can find and use in hugging face). The model with the best performance in STS.
|Model|STS|TC|
|------|---|---|
|mBERT-base|84.66|81.55|
|XML-R-base|89.16|83.52|
|KLUE-RoBERTa-base|92.50|85.07|

[check more ->](https://github.com/KLUE-benchmark/KLUE)

After that, many attempts were made using many models. 
[leand more ->](https://github.com/pokssakk/model-experiments)

<br>

### ➡️ Key Techniques:
1. Sliding window over tokens with max pooling
2. Sentence-level Pertubation + Perplexity filtering
3. KLUE-RoBERTa embedding + cosine similarity for pseudo-labeling
4. Model ensemble using ranking + voting

<br>

### ⭐ Final Model:
#### 1) Data Preparation
Given the competition setting, the original labels were provided only at the Full Text level
- Even if only some paragraphs were AI-generated, the entire text was labeled as 1
- Evaluation, however, required paragraph-level AI probabilities


To address this mismatch, we processed the data as follows:

1. **Paragraph-level Relabeling (KoSimCSE + AutoEncoder)**
  - Each full_text was split into paragraph units
  - For each paragraph, we predicted whether it was likely AI-generated (1) or human-written (0) using two signals:
    - Semantic Similarity (KoSimCSE):
      - Paragraph embeddings (via BM-K/KoSimCSE-roberta-multitask) compared to the corpus mean vector
      - Low cosine similarity → AI-like
    - AutoEncoder-based Reconstruction Error:
      - High AE score → AI-like
  - Threshold:
    - cosine_similarity < μ - 2σ or ae_score > μ + 2σ → labeled as generated=1
  - Title-level Correction:
    - For full_text originally labeled generated=1 but with no positive paragraphs,
      the most suspicious paragraph (highest AE score & lowest similarity) was corrected to 1

2. **Positive Class Augmentation (KANANA)**


<br>

#### 2) Models
Our final ensemble combined two fine-tuned transformer models and a CatBoost classifier:
1. **KLUE-RoBERTa-large (fine-tuned)**  
   - Base weight: `klue/roberta-large`  
   - Chosen for its strong semantic representation capability,
     which is crucial for detecting subtle contextual inconsistencies in AI-generated text
   - [Check training details here]()

2. **KLUE-RoBERTa-base (fine-tuned)**  
   - Base weight: `klue/roberta-base`  
   - Selected as a lighter and more generalizable counterpart to the large model,
     reducing overfitting risk on imbalanced data
   - [Check training details here]()

3. **CatBoost Classifier**  
   - Focused on stylometric anomalies often observed in AI-generated text, complementing semantic models  
   - Features: **unique_ratio**, **verb_ratio**, **entropy**, **polynomial interaction terms (degree=2)**, **Perplexity (PPL)** estimated via `skt/kogpt2-base-v2`  
   - [Check training details here]()

<br>

#### 3) Ensemble strategy
We adopted a custom Extreme Voting to maximize ROC-AUC:
- ≥2 models ≥0.5 → max(probabilities) (optimistic consensus)
- Otherwise → min(probabilities) (conservative fallback)


<br>

## :round_pushpin: Experiments & Results


<br>

## :round_pushpin: Project Structure


<br>

## :round_pushpin: Team & Contributions
Our team of 5 members divided responsibilities as follows: <br>

Team Name: **PokSSak(폭싹)**
|소속|이름|역할||
|------|---|---|---|
|숙명여자대학교 컴퓨터사이언스전공(22)|김소영|||
|숙명여자대학교 데이터사이언스전공(23)|김수빈|||
|숙명여자대학교 데이터사이언스전공(23)|오현서|||
|숙명여자대학교 데이터사이언스전공(23)|원지우|Data Engineering & Experiment Execution||
|숙명여자대학교 소프트웨어융합전공(22)|임소정|Data Engineering & Model Development|[sophia](https://github.com/Sophia680102)|

We collaborated for 4 weeks.


<br>

## :round_pushpin: Stacks
Environment<br>
<img src="https://img.shields.io/badge/googlecolab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white">
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">

Development<br>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

Communication<br>
<img src="https://img.shields.io/badge/notion-000000?style=for-the-badge&logo=notion&logoColor=white">
<img src="https://img.shields.io/badge/googlemeet-00897B?style=for-the-badge&logo=googlemeet&logoColor=white">

<br>

## :round_pushpin: Key Takeaways
- 


<br>

## :round_pushpin: Appendix

- Final submission score: **ROC-AUC 0.889**
- Full competition leaderboard: *[https://dacon.io/competitions/official/236473/leaderboard]*
- PDF summary: 'docs/project_summary.pdf'
