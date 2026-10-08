# Automated Customer Reviews

An NLP system built on Amazon product reviews (Datafiniti dataset). It does four things and combines them in one web app:

1. **Classifies** a review as Negative, Neutral or Positive.
2. **Groups** products into six meta-categories.
3. **Writes** a recommendation article for each category with a generative model.
4. **Visualizes** the reviews in an interactive dashboard (Bonus 1).

```text
Raw reviews ──► Task 1: RoBERTa sentiment classifier ──────────────────────────┐
     │                                                                         │
     └────────► Task 2: product clustering ──┬──► Task 3: Qwen articles ───────┤
                       (6 categories)        │       (one per category)        │
                                             └──► Bonus 1: analytics dashboard ┤
                                                                               ▼
                                                           Task 4: Gradio app (4 tabs)
```

## Notebooks

All notebooks are in [`src/`](src/).

| # | Notebook | What it does | Main result |
|---|---|---|---|
| 1 | [Task_1_Sentiment_Analysis_Train_Val_Test_RoBERTa.ipynb](src/Task_1_Sentiment_Analysis_Train_Val_Test_RoBERTa.ipynb) | Sentiment classification: TF-IDF + Logistic Regression baseline vs. fine-tuned RoBERTa | RoBERTa: **95.6% accuracy, 0.70 macro F1** on the test set |
| 2 | [Task2_clustering.ipynb](src/Task2_clustering.ipynb) | Clusters 92 products into meta-categories using their category tags | **6 categories**; K-Means and Agglomerative agree on 85 of 92 products |
| 3 | [Task_3_Qwen_Recommendation_Improved.ipynb](src/Task_3_Qwen_Recommendation_Improved.ipynb) | Builds evidence per category (rankings, complaints) and has Qwen2.5-1.5B-Instruct write a recommendation article | 6 articles, one per category |
| Bonus 1 | [Bonus_1_Interactive_Review_Analytics.ipynb](src/Bonus_1_Interactive_Review_Analytics.ipynb) | Standalone Plotly + Gradio dashboard: sentiment, product ratings, review lengths, complaint phrases | Interactive dashboard, filterable by category |
| 4 | [Task_4_With_Bonus_1_Visible_Dashboard.ipynb](src/Task_4_With_Bonus_1_Visible_Dashboard.ipynb) | **Final app:** one Gradio app with a tab for each task plus the Bonus 1 dashboard. Also generates `app.py` and `requirements.txt` for deployment | Working 4-tab web app |

**Earlier versions of Task 4,** kept for reference:
- [Task_4_Integrated_Gradio_All_3_Tasks.ipynb](src/Task_4_Integrated_Gradio_All_3_Tasks.ipynb): 3 tabs, without the dashboard.
- [Task_4_With_Bonus_1_Analytics.ipynb](src/Task_4_With_Bonus_1_Analytics.ipynb): the generated `app.py` has 4 tabs, but the app shown inside the notebook still has only 3.

Run the notebooks in order: each one uses files saved by the previous ones.

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

## Bonus 1 — Interactive review analytics

An interactive dashboard built with Plotly and Gradio on top of the Task 2 output (`datasets/clustered_reviews.csv`). It needs no extra data and no extra model.

The dashboard has three controls: a **category** (or "All categories"), a **minimum number of reviews per product**, and **how many products and phrases to show**. For the selected category it displays:

| View | What it shows |
|---|---|
| Summary line | Number of reviews and products, and the average rating |
| Sentiment distribution | Donut chart of negative / neutral / positive reviews |
| Product ratings | The most-reviewed products with their average rating |
| Complaint phrases | The most frequent two-word phrases in negative reviews (e.g. *waste money*, *battery life*, *bad batch*) |
| Review lengths | Histogram of review length in words, split by sentiment |
| Product table | Review count, average rating and % negative per product |

**Note on sentiment:** the dashboard derives sentiment from the **star rating** (same rule as Task 1). It does not run RoBERTa on the reviews. Reviews without a product name are left out, so the dashboard covers 55,611 reviews of 69 products.

The notebook also contains a monthly trend chart. It is skipped because `clustered_reviews.csv` has no review dates.

---

## Task 4 — Integrated Gradio app

A single Gradio app with four tabs:

| Tab | Uses | What the user sees |
|---|---|---|
| **Sentiment Analyzer** | Task 1 model | Type a review and get the predicted sentiment with confidence scores |
| **Category Explorer** | Task 2 output | A summary of all categories; pick one to see its products (ranked by rating) and example reviews |
| **AI Recommendations** | Task 3 output | The generated article for the selected category |
| **Analytics Dashboard** | Task 2 output | The Bonus 1 dashboard |

The app shows the articles that Task 3 already generated. It does not run Qwen for each request, which keeps it light enough for a CPU-only Hugging Face Space.

The notebook also writes a standalone 4-tab `app.py` and its `requirements.txt`. To deploy on a Hugging Face Space, upload these files together:

```text
app.py
requirements.txt
category_recommendation_articles.csv
datasets/clustered_reviews.csv
models/amazon_roberta_sentiment_model/
```

---

## Project structure

```text
ML_automated_customer_reviews/
├── README.md
├── requirements.txt
├── app.py                                   # Gradio app
├── category_recommendation_articles.csv     # Task 3 output
├── src/                                     # notebooks
│   ├── Task_1_Sentiment_Analysis_Train_Val_Test_RoBERTa.ipynb
│   ├── Task2_clustering.ipynb
│   ├── Task_3_Qwen_Recommendation_Improved.ipynb
│   ├── Bonus_1_Interactive_Review_Analytics.ipynb
│   ├── Task_4_With_Bonus_1_Visible_Dashboard.ipynb   # final app
│   ├── Task_4_With_Bonus_1_Analytics.ipynb           # earlier version
│   └── Task_4_Integrated_Gradio_All_3_Tasks.ipynb    # earlier version (3 tabs)
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

**Hardware:** fine-tuning RoBERTa (Task 1) and running Qwen (Task 3) need a GPU. Both notebooks were run on Google Colab. Task 2, Bonus 1 and Task 4 run fine on a CPU.

### Run order
1. **Task 1** → saves the model to `models/amazon_roberta_sentiment_model/`
2. **Task 2** → saves `datasets/clustered_reviews.csv`
3. **Task 3** → saves `category_recommendation_articles.csv`
4. **Bonus 1** (optional) → opens the dashboard on its own
5. **Task 4** (`Task_4_With_Bonus_1_Visible_Dashboard.ipynb`) → starts the 4-tab web app and writes the standalone `app.py`. After that, `python app.py` starts the same app at http://127.0.0.1:7860.

> **Check the paths before running.** Some notebooks expect files in slightly different places:
> - Task 3 reads `clustered_reviews.csv` from the project folder, but Task 2 saves it in `datasets/`.
> - The Task 4 notebook loads the model from `model/` (singular), but Task 1 saves it to `models/` (plural). The generated `app.py` uses the correct `models/` path.
>
> Either copy or rename the files, or change the path at the top of the notebook.

## Limitations

- **Neutral sentiment is hard to detect.** Only 4% of reviews are neutral, and RoBERTa recalls just 28% of them.
- **Product names are unreliable** in `1429_1.csv`: the same name is attached to up to 7 different product ids. Because of this, a few products end up in the wrong category (for example, a Kindle case listed under *Fire Tablets*). Task 2 uses the name only as a secondary signal for this reason.
- **The weighted rating favors products with very few reviews.** With m = 3, products with 3–9 perfect reviews can rank above products with thousands of reviews. A larger m (for example the median review count) would give more robust "top products".
- **Generated articles can include unsupported claims.** Qwen 1.5B sometimes adds features that are not in the evidence, even with the grounding rules. Check the articles against the statistics before relying on them.
- **The dashboard's sentiment comes from star ratings, not from the model.** It shows how customers rated products, not what RoBERTa predicts.
- **"Household, Office & Pets" is a catch-all** for AmazonBasics products. It would split further with k ≥ 7.
