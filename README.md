# Vehicle Class Prediction

A multi-class classification solution for predicting `vehicle_class` from vehicle specifications, built with a restricted set of model families: **Bayesian classification, Naive Bayes, Logistic Regression, and Decision Trees**. Ensembles, boosting, SVMs and neural networks were intentionally not used.

**Final model:** Logistic Regression (L1 penalty, `C=0.1`, balanced class weights) with median imputation and standardization.
**Cross-validated macro-F1:** ~0.898 (local validation only; no official competition score is claimed).

---

## Repository contents

| File | Description |
|---|---|
| `vehicle_pipeline.py` | End-to-end script: inspection, model comparison, tuning, final fit, submission and checks |
| `model_comparison.csv` | Cross-validation results for every candidate model |
| `vehicle_submission.csv` | Final predictions (`vehicle_id`, `vehicle_class`) for the 2,500 test rows |
| `README.md` | This file |

Input data (not included here, add your own copy):
`vehicle_train_10000.csv`, `vehicle_test.csv`, `vehicle_sample_submission.csv`

---

## Dataset

- **Train:** 10,000 rows x 22 columns. **Test:** 2,500 rows x 21 columns.
- **Target:** `vehicle_class`, with 3 classes (0, 1, 2), nearly balanced (33.9% / 33.3% / 32.9%).
- **Features:** 20 numeric specifications, such as engine displacement, horsepower, weight, dimensions, fuel efficiency, cylinders, torque, ground clearance, acceleration, braking distance, tire width, drag coefficient and battery capacity.
- **Data quality:** no missing values in train or test.
- **Train/test compatibility:** the maximum Kolmogorov-Smirnov statistic across features is 0.056, so the distributions are compatible.
- **`vehicle_id`:** treated as an identifier only. Its correlation with the class is 0.002, so it is excluded from the predictors and used only to align the submission.
- **Duplicates:** 2,500 training rows duplicate another row once `vehicle_id` is ignored. They were kept, but they may slightly inflate cross-validation scores.

---

## Method

### Validation
- 80/20 stratified split (`random_state=42`) for model selection and a hold-out check.
- 5-fold stratified cross-validation with identical folds for all models.
- Hyperparameters tuned with `GridSearchCV`, optimizing **macro-F1**.
- All preprocessing sits inside scikit-learn `Pipeline`s, so it is fit only on training folds (no leakage).
- Test data and labels played no part in model selection.

### Candidates

| Family | Implementation | Tuned parameters |
|---|---|---|
| Logistic Regression | `LogisticRegression` + `StandardScaler` | `C`, L1/L2 penalty, class weight (`saga` solver) |
| Naive Bayes | `GaussianNB` | `var_smoothing` |
| Bayesian (distinct) | `QuadraticDiscriminantAnalysis` | `reg_param` |
| Decision Tree | `DecisionTreeClassifier` | `criterion`, `max_depth`, `min_samples_leaf`, `ccp_alpha` |

**Bayesian note:** Gaussian Naive Bayes is Bayes' theorem with a conditional-independence assumption between features. QDA is a Bayesian classifier that models a full covariance matrix per class, so it does not make that assumption. That makes it a meaningfully different candidate, not a duplicate.

No feature engineering or correlated-feature removal was applied. Scaling was used only for Logistic Regression.

### Selection rule
The model with the highest cross-validated macro-F1 wins. If candidates fall within 0.002 of the best, the simpler model is preferred.

---

## Results (5-fold CV on the 80% training split)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc. | Std (macro-F1) |
|---|---|---|---|---|---|
| **Logistic Regression, tuned** | 0.8978 | **0.8981** | 0.8976 | 0.8982 | 0.0046 |
| Logistic Regression, untuned | 0.8956 | 0.8960 | 0.8956 | 0.8961 | 0.0066 |
| QDA, tuned | 0.8923 | 0.8929 | 0.8925 | 0.8926 | 0.0022 |
| QDA, untuned | 0.8901 | 0.8909 | 0.8904 | 0.8904 | 0.0026 |
| Decision Tree, tuned | 0.8871 | 0.8877 | 0.8873 | 0.8875 | 0.0080 |
| Decision Tree, untuned | 0.8849 | 0.8853 | 0.8848 | 0.8853 | 0.0034 |
| Gaussian NB, tuned | 0.8460 | 0.8470 | 0.8465 | 0.8465 | 0.0067 |
| Gaussian NB, untuned | 0.8451 | 0.8462 | 0.8456 | 0.8456 | 0.0063 |

Full numbers, best parameters and run times are in `model_comparison.csv`.

**Interpretation**
- Logistic Regression is best, but the gaps to untuned Logistic Regression and to QDA are within about one fold standard deviation. Tuning gave only a small gain.
- Gaussian Naive Bayes is clearly weakest, consistent with its independence assumption fitting poorly on correlated vehicle specifications.
- For the tuned tree, the search chose no pruning (`ccp_alpha=0`, unlimited depth), but it still trails the linear and QDA models.

### Hold-out check (20% split, model refit on the other 80%)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 0.832 | 0.820 | 0.826 |
| 1 | 0.892 | 0.902 | 0.897 |
| 2 | 0.920 | 0.922 | 0.921 |

Accuracy 0.881, macro-F1 0.881.

Confusion matrix (rows = true, columns = predicted):

```
[[556  72  50]
 [ 62 600   3]
 [ 50   1 606]]
```

Class 0 is the hardest and is confused with both other classes. Classes 1 and 2 are almost never confused with each other.

---

## Final submission

The winning pipeline was refit on all 10,000 labeled rows and used to predict class labels (not probabilities) for the test set.

Verified automatically in the script:
- 2,500 rows
- columns exactly `vehicle_id`, `vehicle_class`, in that order, with no index column
- IDs match `vehicle_sample_submission.csv` in the same order
- all predicted labels are classes seen in training

Predicted class counts: 826 / 838 / 836 for classes 0 / 1 / 2.

---

## How to run

```bash
pip install pandas numpy scikit-learn scipy
python vehicle_pipeline.py
```

The script currently reads data from `/mnt/user-data/uploads/` and writes to `/mnt/user-data/outputs/`. Change the `D` and `O` path variables near the top of `vehicle_pipeline.py` to match your local folders (for example `D = 'data/'` and `O = './'`). The full run takes a couple of minutes, mostly in the Logistic Regression and Decision Tree grid searches.

---

## Limitations

- All scores are local cross-validation or hold-out estimates, not official competition scores. Treat the leaderboard score as the true test.
- Duplicate rows may make cross-validation slightly optimistic.
- The hold-out accuracy (0.881) is a little below the cross-validation figure (0.898), likely from sampling noise and mild optimism from tuning on the same data.
- Differences between the top models are small relative to fold variability, so the ranking of Logistic Regression versus QDA is not decisive.
- Feature engineering and interaction terms were not explored and could improve the linear and Bayesian models.
