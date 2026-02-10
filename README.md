# Intent Classification for Support Ticket Routing

## What this project does
This project builds a lightweight NLP model that routes incoming customer support messages to the correct **intent queue** (27 intents).  
Example intents include: `cancel_order`, `get_refund`, `recover_password`, `track_order`, `delete_account`.

## Dataset
**Bitext Sample Customer Service Dataset** (CSV files with predefined splits)

- Training: **6,539** rows
- Test: **818** rows

Columns present: `utterance`, `intent`, `category`, `tags`  
This project uses:
- **Text**: `utterance`
- **Label**: `intent`

## Method
### Preprocessing (minimal and reproducible)
- Lowercasing
- Replace URLs with `__url__`
- Replace emails with `__email__`
- Whitespace normalization

### Model
- **TF-IDF** vectorizer with 1–2 grams
- **Logistic Regression** classifier (`class_weight="balanced"`)

## Results
### Validation
- Accuracy: **0.996**
- Macro F1: **0.996**

### Test
- Accuracy: **0.996**
- Macro F1: **0.996**
- Test set misclassifications: **3 / 818**

### Confusion matrix
The confusion matrix is strongly diagonal (near-perfect routing).  
Saved as:
- `outputs/confusion_matrix_test.png`

### Quick error analysis (examples)
The few errors were boundary cases:
- `delete_account` → `create_account` for: “I do not know how I can cancel an online account”
- `get_refund` → `track_refund` for: “I need a refund” (very short / ambiguous)
- `track_order` → `delivery_period` for: “when will my order arrive” (overlapping intent boundary)

## Confidence-based manual triage (human-in-the-loop)
To reduce risk on ambiguous messages, route low-confidence predictions to manual review:

- Threshold used: **0.60** on the max predicted probability
- Test set manual review rate at 0.60: **8.68%**
- Auto-routed rate: **91.32%**

In a real setting, this threshold is tuned based on the cost of misrouting and the triage team’s capacity.

## How to run

### 1) Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### 2) Train + evaluate
Run the notebook:
- `notebooks/01_baseline_tfidf.ipynb`

### 3) Inference demo (CLI)
After you save the trained pipeline to:
- `models/intent_router.joblib`

Run:
```bash
python src/predict.py
```

The script prints top-k predicted intents with confidence scores.

## Repo structure
```
support_ticket_nlp/
  data/
    Bitext_Sample_Customer_Service_Training_Dataset.csv
    Bitext_Sample_Customer_Service_Validation_Dataset.csv
    Bitext_Sample_Customer_Service_Testing_Dataset.csv
  notebooks/
    01_baseline_tfidf.ipynb
  src/
    predict.py
  models/
    intent_router.joblib
  outputs/
    confusion_matrix_test.png
  README.md
```

## Monitoring plan (what to track in production)
- Input drift: message length and vocabulary shift
- Confidence drift: distribution of max predicted probability
- Intent distribution drift: spikes in certain intents (refunds, cancellations)
- Manual triage rate and outcomes (used as feedback labels for retraining)

## Notes
This dataset is curated and cleaner than many real-world support logs. In production, performance can drop due to noisier text, mixed intents, and evolving user language. Confidence-based triage + monitoring helps maintain reliability over time.
