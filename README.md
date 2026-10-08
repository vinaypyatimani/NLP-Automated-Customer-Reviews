# Automated Customer Reviews

An NLP system built on Amazon product reviews (Datafiniti dataset). It does three things and combines them in one web app:

1. **Classifies** a review as Negative, Neutral or Positive.
2. **Groups** products into six meta-categories.
3. **Writes** a recommendation article for each category with a generative model.

```text
Raw reviews ──► Task 1: RoBERTa sentiment classifier ─────────────────────┐
     │                                                                    │
     └────────► Task 2: product clustering ──► Task 3: Qwen articles ─────┤
                         (6 categories)          (one per category)       ▼
                                                              Task 4: Gradio app (3 tabs)
```

## Notebooks

| # | Notebook | What it does | Main result |
|---|---|---|---|
| 1 | [src/Task_1_Sentiment_Analysis_Train_Val_Test_RoBERTa.ipynb](src/Task_1_Sentiment_Analysis_Train_Val_Test_RoBERTa.ipynb) | Sentiment classification: TF-IDF + Logistic Regression baseline vs. fine-tuned RoBERTa | RoBERTa: **95.6% accuracy, 0.70 macro F1** on the test set |
| 2 | [src/Task2_clustering.ipynb](src/Task2_clustering.ipynb) | Clusters 92 products into meta-categories using their category tags | **6 categories**; K-Means and Agglomerative agree on 85 of 92 products |
| 3 | [src/Task_3_Qwen_Recommendation_Improved.ipynb](src/Task_3_Qwen_Recommendation_Improved.ipynb) | Builds evidence per category (rankings, complaints) and has Qwen2.5-1.5B-Instruct write a recommendation article | 6 articles, one per category |
| 4 | [src/Task_4_Integrated_Gradio_All_3_Tasks.ipynb](src/Task_4_Integrated_Gradio_All_3_Tasks.ipynb) | Gradio app with one tab per task, plus a script that generates the deployment files | Working 3-tab web app |

Run them in order: each notebook uses files saved by the previous ones.

---

## Task 1 — Sentiment classification

**Labels** come from the star rating: 1–2 = negative, 3 = neutral, 4–5 = positive. The model only sees the text (review body + title).

**Data:** `1429_1.csv`, 34,620 reviews after removing missing values. The classes are very imbalanced: 93.3% positive, 4.3% neutral, 2.3% negative.

**Split (stratified):** 24,234 train / 7,270 validation / 3,116 test.

**Models:**
- **Baseline:** TF-IDF (unigrams + bigrams, 50k features) + Logistic Regression with `class_weight="balanced"`.
- **Transformer:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), fine-tuned for up to 3 epochs (lr 2e-5, max 256 tokens). The best epoch is chosen by validation macro F1, with early stopping.

**Test results:**

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.930 | 0.600 | 0.612 | 0.606 |
| Fine-tuned RoBERTa | **0.956** | **0.749** | **0.694** | **0.696** |

Per-class F1 for RoBERTa: negative 0.72, neutral 0.39, positive 0.98. Fine-tuning helps most on the negative class (F1 0.50 → 0.72). **Neutral reviews remain the hardest class**: their recall is only 28%, because most of them are predicted as positive.

Accuracy is misleading here: always predicting "positive" would already give 93%. That is why macro F1 is used to choose and compare models.

**Output:** `models/amazon_roberta_sentiment_model/` (model weights + tokenizer).

---

## Task 2 — Product category clustering

**Data:** all three CSV files combined. That gives 67,992 rows; after removing 7,151 reviews that appear in more than one file, **60,841 reviews of 92 products** remain.

**Why cluster products, not reviews:** categories belong to a product, so every review of a product should get the same category. Clustering 92 products is fast and gives each product equal weight.

**Pipeline:**
1. **Clean the `categories` tags:** remove store and marketing tags (*Frys*, *Featured Brands*, *See more…*), internal codes and brand words. Unify spelling variants (*eBook Readers*, *E-Readers* → `ereader`) and turn plurals into singular.
2. **TF-IDF** over a vocabulary built from the cleaned categories (~390 unigrams and bigrams). The product name is projected onto the same vocabulary at weight 0.5.
3. **LSA** (TruncatedSVD, 10 components), so that related terms end up close together.
4. **K-Means** and **Agglomerative (Ward)**, tested for k = 3–10. **k = 6** has the best silhouette and Davies-Bouldin scores within the target range of 4–6 categories.
5. **Name the clusters** automatically by matching each cluster to keyword lists (Hungarian algorithm).

**Categories (K-Means):**

| Category | Products | Reviews |
|---|---|---|
| Fire Tablets | 27 | 29,635 |
| Household, Office & Pets | 15 | 12,140 |
| Smart Speakers & Echo | 12 | 8,737 |
| Streaming & TV | 7 | 5,087 |
| E-readers & Kindle | 18 | 5,064 |
| Chargers & Cables | 13 | 178 |

**Model comparison:**

| | K-Means | Agglomerative (Ward) |
|---|---|---|
| Silhouette (higher is better) | **0.607** | 0.596 |
| Davies-Bouldin (lower is better) | **1.130** | 1.148 |
| Stability (mean ARI on 80% subsamples) | **0.915** | 0.911 |

The two models agree closely (ARI 0.87). All 7 products they disagree on fall between *E-readers & Kindle* and *Chargers & Cables*. **K-Means is used downstream**: it scores slightly better, places the disputed Kindle e-reader correctly, and can assign new products with `predict`.

**Output:** `datasets/clustered_reviews.csv`, which contains every review with its `kmeans_cluster` and `kmeans_category`.

---

## Task 3 — Generative recommendation articles

Statistics and classic NLP gather the facts first. The language model then only turns those facts into readable text.

```text
Task 2 categories → product statistics → weighted ranking → negative reviews
→ TF-IDF complaint themes → evidence package → grounded prompt → Qwen2.5-1.5B-Instruct → article
```

1. **Product statistics** per category: review count, average rating, % positive / neutral / negative.
2. **Weighted (Bayesian) rating:** `(v/(v+m))·R + (m/(v+m))·C`. This pulls products with few reviews toward the overall average (C = 4.54).
3. **Complaint themes:** TF-IDF terms from each category's negative reviews (e.g. *slow, apps, screen* for Fire Tablets; *refurbished, boot* for Streaming & TV; *dead, long* for batteries), plus the longest negative reviews as examples.
4. **Grounded prompt:** the evidence plus strict rules (use only the supplied evidence, never invent specifications). The article must have six fixed sections: Overview, Top 3 Products, Key Differences, Common Complaints, Weakest Product, Final Recommendation.
5. **Generation** with [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (temperature 0.3, up to 700 new tokens).

**Outputs:** `category_recommendation_articles.csv` and `product_recommendation_articles.md`.

---

## Task 4 — Integrated Gradio app

A single Gradio app with three tabs:

| Tab | Uses | What the user sees |
|---|---|---|
| **Sentiment Analyzer** | Task 1 model | Type a review and get the predicted sentiment with confidence scores |
| **Category Explorer** | Task 2 output | A summary of all categories; pick one to see its products (ranked by rating) and example reviews |
| **AI Recommendations** | Task 3 output | The generated article for the selected category |

The app shows the articles that Task 3 already generated. It does not run Qwen for each request, which keeps it light enough for a CPU-only Hugging Face Space.

The notebook also creates a standalone `app.py` and `requirements.txt`, and uploads them to the Hugging Face Hub.

---

## Project structure

```text
ML_automated_customer_reviews/
├── README.md
├── requirements.txt
├── app.py                                   # Gradio app
├── category_recommendation_articles.csv     # Task 3 output
├── src/                                     # notebooks for Tasks 1–4
│   ├── Task_1_Sentiment_Analysis_Train_Val_Test_RoBERTa.ipynb
│   ├── Task2_clustering.ipynb
│   ├── Task_3_Qwen_Recommendation_Improved.ipynb
│   └── Task_4_Integrated_Gradio_All_3_Tasks.ipynb
├── datasets/                                # not tracked in git
│   ├── 1429_1.csv
│   ├── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
│   ├── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
│   └── clustered_reviews.csv                # Task 2 output
└── models/                                  # not included; created by Task 1
    └── amazon_roberta_sentiment_model/
```

The notebooks in `src/` use file paths relative to the project folder (for example `datasets/1429_1.csv`). Run them with the project folder as the working directory. Alternatively, add `%cd ..` as the first cell when running them locally.

## Setup

### Data
CSV files are not tracked in git. Download the [Datafiniti Amazon consumer reviews dataset](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) and put the three CSV files in `datasets/`.

### Environment
```bash
pip install -r requirements.txt
```

**Hardware:** fine-tuning RoBERTa (Task 1) and running Qwen (Task 3) need a GPU. Both notebooks were run on Google Colab. Tasks 2 and 4 run fine on a CPU.

### Run order
1. **Task 1** → saves the model to `models/amazon_roberta_sentiment_model/`
2. **Task 2** → saves `datasets/clustered_reviews.csv`
3. **Task 3** → saves `category_recommendation_articles.csv`
4. **Task 4** → starts the 3-tab web app and writes the standalone `app.py`. After that, `python app.py` starts the same app at http://127.0.0.1:7860.

> **Check the paths before running.** Some notebooks expect files in slightly different places:
> - Task 3 reads `clustered_reviews.csv` from the project folder, but Task 2 saves it in `datasets/`.
> - Task 4 loads the model from `model/` (singular), but Task 1 saves it to `models/` (plural).
>
> Either copy or rename the files, or change the path at the top of the notebook.

## Limitations

- **Neutral sentiment is hard to detect.** Only 4% of reviews are neutral, and RoBERTa recalls just 28% of them.
- **Product names are unreliable** in `1429_1.csv`: the same name is attached to up to 7 different product ids. Because of this, a few products end up in the wrong category (for example, a Kindle case listed under *Fire Tablets*). Task 2 uses the name only as a secondary signal for this reason.
- **The weighted rating favors products with very few reviews.** With m = 3, products with 3–9 perfect reviews can rank above products with thousands of reviews. A larger m (for example the median review count) would give more robust "top products".
- **Generated articles can include unsupported claims.** Qwen 1.5B sometimes adds features that are not in the evidence, even with the grounding rules. Check the articles against the statistics before relying on them.
- **"Household, Office & Pets" is a catch-all** for AmazonBasics products. It would split further with k ≥ 7.
