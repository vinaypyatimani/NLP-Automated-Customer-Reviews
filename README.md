# Automated Customer Reviews

NLP project on Amazon product reviews (Datafiniti dataset).

## Notebooks

| Notebook | Task | Approach | Result |
|---|---|---|---|
| [Task_1_Sentiment_Analysis_Amazon_Reviews.ipynb](Task_1_Sentiment_Analysis_Amazon_Reviews.ipynb) | Classify reviews as **Negative** (1–2★), **Neutral** (3★) or **Positive** (4–5★) | TF-IDF + Logistic Regression (`class_weight="balanced"`) | ~93% test accuracy |
| [Task2_product_clustering.ipynb](Task2_product_clustering.ipynb) | Group reviews into 4–6 product **meta-categories** | Category-word TF-IDF + K-Means (k = 4), plus hyperparameter tuning | 4 categories: E-readers, Fire Tablets, Smart Home & Speakers, Streaming & Media |

## Data

CSV files are not tracked in git. Download the [Datafiniti Amazon reviews dataset](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) and place the files in `datasets/`:

- Task 1 uses `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`
- Task 2 uses `1429_1.csv` and writes `clustered_amazon_reviews.csv`

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run the notebooks from this folder so the `datasets/` paths resolve.
