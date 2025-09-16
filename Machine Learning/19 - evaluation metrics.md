Resource - https://www.notion.so/26d68bea3447808c90f5e29ca3300f26?v=26d68bea344780468bc1000c605fe876

# 📊 Evaluation Metrics for Machine Learning

Choosing the right evaluation metric is as important as choosing the right model.  
This guide covers metrics for **Regression** and **Classification**, including when to use and when to avoid them.

---

## 📈 Regression Metrics

### 1. Mean Squared Error (MSE)
**Formula:**
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

**Purpose:**  
- Measures average squared difference between predicted and actual values.  
- Penalizes **large errors more heavily** (because of squaring).  

**When to use:**  
- When large errors are **very costly** (e.g., predicting medical dosage, financial forecasting).  

**When NOT to use:**  
- If your application is tolerant to large outliers (MSE will exaggerate them).  

---

### 2. Root Mean Squared Error (RMSE)
**Formula:**
\[
RMSE = \sqrt{MSE}
\]

**Purpose:**  
- Same as MSE, but in the **same units as the target variable**.  
- Easier to interpret (e.g., "average prediction error is $5,000").  

**When to use:**  
- Same as MSE, but especially when stakeholders prefer interpretable error units.  

**When NOT to use:**  
- Same caveats as MSE — sensitive to outliers.  

---

### 3. Mean Absolute Error (MAE)
**Formula:**
\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

**Purpose:**  
- Average magnitude of errors, without squaring.  
- Treats all errors equally (linear penalty).  

**When to use:**  
- When you want a **robust metric** less sensitive to outliers.  
- Example: Predicting delivery times. A 30-min error is bad, but not “10x worse” than a 3-min error.  

**When NOT to use:**  
- If large deviations are **critical** (then prefer MSE/RMSE).  

---

### 4. Mean Absolute Percentage Error (MAPE)
**Formula:**
\[
MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
\]

**Purpose:**  
- Expresses errors as percentages.  
- Easy for business stakeholders to understand (e.g., "on average, predictions are 8% off").  

**When to use:**  
- When actual values are always **positive** and non-zero.  

**When NOT to use:**  
- If \(y_i = 0\) or very close to 0 → division explodes.  
- If you care more about absolute units (e.g., predicting $ instead of %).  

---

### 5. Mean Relative Error (MRE)
**Formula:**
\[
MRE = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{|y_i|}
\]

**Purpose:**  
- Similar to MAPE but doesn’t multiply by 100.  
- Useful when relative errors are more meaningful than absolute ones.  

**When to use:**  
- Same as MAPE, but when you want ratio values instead of %.  

**When NOT to use:**  
- Same limitations as MAPE (can blow up at small denominators).  

---

## 📊 Classification Metrics

### 1. Accuracy
**Formula:**
\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

**Purpose:**  
- Fraction of correct predictions.  

**When to use:**  
- When classes are **balanced** (roughly equal positives and negatives).  

**When NOT to use:**  
- When data is **imbalanced** (e.g., 99% non-fraud vs 1% fraud → 99% accuracy but useless).  

---

### 2. Precision
**Formula:**
\[
Precision = \frac{TP}{TP + FP}
\]

**Purpose:**  
- Of all predicted positives, how many are actually positive.  
- "How many emails I flagged as spam were really spam?"  

**When to use:**  
- When **false positives are costly**.  
- Example: Sending fraud alerts → don’t annoy users with too many false alarms.  

**When NOT to use:**  
- If false negatives matter more (then use Recall).  

---

### 3. Recall (Sensitivity / True Positive Rate)
**Formula:**
\[
Recall = \frac{TP}{TP + FN}
\]

**Purpose:**  
- Of all actual positives, how many were correctly identified.  
- "How many spam emails did I catch out of all spam emails?"  

**When to use:**  
- When **false negatives are costly**.  
- Example: Cancer detection → missing a positive case is dangerous.  

**When NOT to use:**  
- If false positives are more costly (then prefer Precision).  

---

### 4. F1-Score
**Formula:**
\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

**Purpose:**  
- Harmonic mean of Precision and Recall.  
- Balances the two when you can’t prioritize one over the other.  

**When to use:**  
- Imbalanced datasets, when both FP and FN matter.  

**When NOT to use:**  
- If your problem strongly prefers Precision *or* Recall.  

---

### 5. Log Loss (Binary Cross-Entropy)
**Formula:**
\[
LogLoss = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
\]

**Purpose:**  
- Measures how well predicted probabilities match actual labels.  
- Penalizes confident but wrong predictions heavily.  

**When to use:**  
- When you need **probabilistic outputs** (e.g., risk scoring, medical tests).  

**When NOT to use:**  
- If you only care about hard class labels (then Accuracy/Precision/Recall is enough).  

---

### 6. Cross-Validation
**Purpose:**  
- Not a metric, but a **technique** to evaluate models.  
- Splits data into training/validation folds and tests model multiple times.  
- Provides a more reliable estimate of generalization.  

**When to use:**  
- Always recommended unless dataset is huge (then train/val split may suffice).  

---

## 🧭 Metric Selection Guide

- **Regression**
  - If large errors are very costly → **MSE / RMSE**.  
  - If robust to outliers → **MAE**.  
  - If stakeholders want relative errors → **MAPE / MRE**.  

- **Classification**
  - Balanced classes → **Accuracy**.  
  - Imbalanced classes:
    - False positives costly → **Precision**.  
    - False negatives costly → **Recall**.  
    - Both matter → **F1-score**.  
  - Need probability calibration → **Log Loss**.  

- **Always** use **Cross-Validation** to confirm model robustness.  

---

## 🚀 Quick Analogies

- **MSE vs MAE**:  
  MSE is like saying “a 10x mistake is 100x worse” → harsh.  
  MAE says “a 10x mistake is 10x worse” → fair.  

- **Precision vs Recall**:  
  Precision = "Of all the alarms I raised, how many were real fires?"  
  Recall = "Of all real fires, how many alarms did I raise?"  

- **F1-score**: A peace treaty between Precision & Recall.  

---
