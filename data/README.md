# Bundled real Olist feature table

`olist_orders.csv.gz` is a derived **real**, historical order-level dataset, not a
synthetic fixture. It ships with the student package so no Kaggle account, network,
raw seven-table database, or teacher directory is required at runtime.

- Original: **Brazilian E-Commerce Public Dataset by Olist**, Olist and collaborators.
- Source identifier: **`olistbr/brazilian-ecommerce`** (Kaggle).
- Source landing page: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Original and this derivative: **CC BY-NC-SA 4.0**. See `LICENSE.md`.
- Exact compressed-file SHA-256, row count, date range, cutoff counts, column order,
  and hashes of the six actual local source CSVs are recorded in `metadata.json`.

## Reproduction (maintainers only)

From repository root, with pandas installed:

```sh
python teacher/build_local_dataset.py
```

This reads only the already-present `teacher/olist_srv/analyst_export/{orders,
order_items,payments,customers}.csv` and `teacher/olist_srv/dims/{products,sellers}.csv`.
It does not download or replace the source. The analyst export contains purchases
before 2017-08-28. Keep delivered orders with observed purchase, actual delivery,
and estimated delivery timestamps; inner-join order items. There is no sampling.
The CSV is sorted by purchase timestamp and order ID. A fixed gzip mtime and empty
gzip filename make rebuilds deterministic in the pinned Python/pandas environment.

The existing aggregation is preserved:

| Field | Meaning / availability |
|---|---|
| `order_id` | Original unique order ID; **not a model feature** |
| `order_purchase_timestamp` | Purchase time in original local-time notation, ISO string; used only for chronological splitting |
| `price` | Sum of item prices |
| `freight` | Sum of freight charges |
| `weight` | Sum of item product weights (grams); pandas sum skips missing weights, all-missing becomes 0 |
| `n_items` | Number of order-item records |
| `installments` | Maximum payment installments per order; absent payment becomes 0 |
| `est_days` | Floor of promised delivery timestamp minus purchase timestamp, in days |
| `customer_state` | Customer state at purchase |
| `seller_state` | State of first seller in original item order |
| `category` | First nonmissing product category in original item order |
| `payment_type` | First nonmissing payment type in original payment order |
| `late` | **Target only**: actual delivery calendar date strictly after promised estimated delivery date (0/1) |

Missing categorical fields become the explicit string `unknown`. No reviews,
review scores, carrier dates, actual delivery dates, or other post-purchase fields
are released as features. Normalizing actual delivery to a calendar date avoids
labeling a delivery later on its promised day as late. This retains the original
prototype's target convention.

## Evaluation limits

The fixed split is train purchases `< 2017-07-31 00:00:00`, validation purchases
`>= 2017-07-31 00:00:00`; both sides contain both classes. This is a retrospective,
delivered-order cohort with outcomes already observed. It excludes canceled or
undelivered orders and is not an unbiased live deployment sample. Original timestamps
have no timezone; the helper interprets them consistently as UTC for comparisons,
without claiming the original event timezone was UTC. Dataset preparation/aggregation
is supplied: students implement pipeline stages, not raw joins or the model algorithm.

`src/olist_model.py` fits numerical mean/std and one-hot vocabularies on **training
rows only**, rejects invalid/nonfinite numerical data and unusable splits, and
supports unseen categorical values. `validation_pr_auc` is average precision.
Class-weighted training means sigmoid scores are not calibrated delivery risks.
