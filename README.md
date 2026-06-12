# Recommendation Systems for Personalized Content Discovery

A complete movie recommendation system built on the **Netflix Prize Dataset**, implementing and comparing multiple approaches including Matrix Factorization (Funk-SVD) and Item-Based Collaborative Filtering.

---

## Results Summary

| Model | RMSE | MAP@10 |
|---|---|---|
| Global Mean | 1.0540 | — |
| User Mean | 0.9810 | — |
| Item Mean | 0.9825 | — |
| Item-CF | 0.8809 | — |
| **Funk-SVD** | **0.8151** | **0.0153** |

Funk-SVD achieves a **22.6% RMSE improvement** over the global mean baseline and converges in 15 epochs with early stopping (best val RMSE: 0.7897).

---

## Repository Structure

```
netflix-recsys/
├── data/
│   └── README.md                  # Instructions to download the dataset
├── notebooks/
│   └── netflix_recsys.ipynb       # Main Kaggle notebook (full pipeline)
├── src/
│   ├── training_pipeline.py       # Complete end-to-end pipeline (all functions)
│   ├── matrix_factorization.py    # Funk-SVD model definition and training
│   ├── item_cf.py                 # Item-Based CF similarity and prediction
│   ├── baselines_and_eval.py      # Baselines, RMSE, train/test split
│   ├── reco_pipeline.py           # Recommendation generation (4-stage pipeline)
│   ├── hit_rate.py                # Hit Rate@K evaluation
│   ├── map_evaluation.py          # MAP@K with sampled negatives
│   └── report_data_collector.py   # Collects all metrics for reporting
├── eda_plots/
│   ├── 01_user_activity.png
│   ├── 02_movie_popularity.png
│   ├── 03_rating_distribution.png
│   ├── 04_temporal_trend.png
│   ├── 05_long_tail.png
│   └── 06_release_year.png
├── reports/
│   └── netflix_report.docx        # Technical report (9 sections, <10 pages)
├── presentation/
│   └── netflix_presentation.pptx  # Presentation slides (8 slides)
├── requirements.txt
└── README.md
```

---

## Dataset

**Netflix Prize Dataset** — available on Kaggle:
```
https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
```

| Property | Value |
|---|---|
| Total Ratings | 100,480,507 |
| Users | 480,189 |
| Movies | 17,770 |
| Rating Scale | 1–5 stars |
| Date Range | November 1999 – December 2005 |
| Subset Used | 39,999,477 ratings (41,335 most active users) |

Download the dataset and place the parquet/CSV files at the paths specified in the config section of `training_pipeline.py`.

---

## Setup

### Requirements

```bash
pip install torch numpy pandas scipy matplotlib pyarrow
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### GPU (recommended)

The training pipeline is designed for **GPU acceleration**. Training Funk-SVD and building Item-CF similarity on GPU (e.g. Kaggle T4) reduces runtime from hours to minutes.

- Funk-SVD training: ~5–10 minutes on T4
- Item-CF similarity build: ~10–20 minutes on T4

---

## Reproducing Results

### Option 1 — Kaggle Notebook (recommended)

1. Open `notebooks/netflix_recsys.ipynb` on Kaggle
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Add the Netflix Prize dataset to the notebook inputs
4. Run all cells in order

### Option 2 — Local Script

Edit the config paths at the top of `training_pipeline.py`:

```python
RATINGS_PATH = "path/to/ratings_df.parquet"
MOVIES_PATH  = "path/to/movies.csv"
```

Then run:

```python
python src/training_pipeline.py
```

---

## Pipeline Overview

The pipeline follows five stages:

### 1. Data Loading and Subsetting
Ratings are loaded from parquet, columns downcast for memory efficiency, and the dataset is filtered to the top 41,335 most active users (~40M ratings).

### 2. Train / Test Split
Time-based per-user split: each user's most recent 20% of ratings form the test set. Users with fewer than 5 ratings are kept fully in training. This mirrors the real-world task of predicting future preferences from past behaviour.

### 3. Model Training

**Baselines** (Global Mean, User Mean, Item Mean) — fit in seconds on CPU.

**Item-CF** — adjusted cosine similarity computed block-wise on GPU. Top-100 neighbours stored per movie. Predictions are weighted sums over a user's rated neighbours.

**Funk-SVD** — bias-aware matrix factorization:

```
r_hat(u,i) = mu + b_u + b_i + p_u · q_i
```

Trained with mini-batch Adam, 50 latent factors, learning rate 0.01, regularisation 0.02, early stopping patience 3.

### 4. Recommendation Generation (4 stages)
1. **Candidate Generation** — all movies not yet rated by the user
2. **Rating Prediction** — score each candidate with the trained MF model
3. **Ranking** — sort by predicted score descending
4. **Top-K Selection** — return top 10 with titles and predicted scores

### 5. Evaluation
- **RMSE** — computed on all held-out (user, movie, rating) triplets
- **MAP@10** — standard AP@K formula; a movie is relevant if actual rating >= 4.0
- **Hit Rate@10** — fraction of users with at least one relevant movie in their top-10

---

## Key Findings

- **Funk-SVD** achieves the best RMSE (0.8151) and MAP@10 (0.0153), representing a 22.6% RMSE reduction over the trivial baseline.
- **Item-CF** provides a strong complementary model (RMSE 0.8809) and is more interpretable.
- **Popularity bias** is a key challenge: the top 20% of movies hold 89.2% of all ratings, pushing models towards popular titles.
- **Eclectic-taste users** (e.g. users who rate across every genre) produce the clearest failure cases — their averaged latent vectors fail to capture any specific preference theme.
- A **hybrid blend** of MF and Item-CF scores consistently outperforms either model alone on RMSE.

---

## Evaluation Protocol

MAP@10 is computed using the standard Average Precision formula:

```
AP@10(u) = (1 / min(10, |relevant_u|)) * sum_{k=1}^{10} P(k) * rel(k)
```

where `rel(k) = 1` if the item at rank k has actual rating >= 4.0, and `P(k)` is precision in the top-k. MAP@10 is the mean of AP@10 across all eligible users.

**Relevance threshold: actual rating >= 4.0**

---

## Project Requirements Coverage

| Requirement | Status |
|---|---|
| EDA (user activity, popularity, rating dist., sparsity, temporal, long-tail) | ✅ |
| At least one recommendation model | ✅ Funk-SVD |
| At least two approaches compared | ✅ Baselines + Item-CF + Funk-SVD |
| Top-K recommendation generation | ✅ 4-stage pipeline |
| RMSE evaluation | ✅ |
| MAP@10 evaluation | ✅ |
| Train/test split methodology stated | ✅ Time-based per-user |
| Sample recommendations with success/failure analysis | ✅ |
| Technical report (≤10 pages) | ✅ |
| Presentation (≤8 slides) | ✅ |
| GitHub repository with documentation | ✅ |

---

## Future Work

- **Neural Collaborative Filtering** — replace the dot product with a multi-layer neural network
- **Temporal dynamics** — TimeSVD++ to capture how user taste evolves over time
- **Content-based features** — genre, director, cast metadata to address cold-start
- **Hybrid ensemble** — weighted blend of MF, CF, and content signals
- **A/B testing** — deploy as an API and evaluate with online metrics

---

## References

- Funk, S. (2006). *Netflix Update: Try This at Home*. Simon Funk Blog.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30–37.
- Netflix Prize Dataset. Kaggle. https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook* (2nd ed.). Springer.
