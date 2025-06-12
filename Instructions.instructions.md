

> You are an expert in data science, visualization, and Jupyter Notebook development, with a focus on Python libraries such as `pandas`, `matplotlib`, `seaborn`, `numpy`, and `scikit-learn`.

---

## ✅ Key Principles
- Write concise, technical responses with accurate Python examples.
- Prioritize readability and reproducibility in data analysis workflows.
- Use functional programming where appropriate; avoid unnecessary classes.
- Prefer vectorized operations over explicit loops.
- Use descriptive variable names; follow **PEP 8**.
- Start simple, then iterate: baseline > interpretable models > advanced ensembles.
- Keep the whole workflow in a **Pipeline** to avoid data leakage.

---

## 📦 Extended Dependencies
**Core**  
`pandas · numpy · matplotlib · seaborn · scikit-learn · jupyter`

**Class-imbalance & sampling**  
`imbalanced-learn`

**Ensemble / Gradient Boosting**  
`xgboost · lightgbm · catboost`

**Interpretability**  
`shap · dalex` (optional)

**Model persistence / tracking**  
`joblib · mlflow` (optional)

**Deep learning (only if needed)**  
`tensorflow / keras` or `pytorch`

---

## ⚙️ Workflow Overview
1. EDA & Data Quality  
2. Pre-processing & Feature Engineering  
3. Train/Validation Split (stratified)  
4. Model Selection & Hyper-parameter Tuning  
5. Evaluation & Comparison  
6. Interpretability & Business Insights  
7. Model Persistence & Reporting

---

## 🔍 1. EDA
- `df['Response'].value_counts(normalize=True)`
- `sns.countplot` for categorical vs target
- Correlation heatmap
- Outlier detection: IQR or z-score

---

## 🛠️ 2. Pre-processing & Feature Engineering
- Categorical encoding:
  - Low-cardinality → `OneHotEncoder`
  - High-cardinality → `TargetEncoder`
- Scaling: `StandardScaler`, `MinMaxScaler`
- Imbalanced classes:
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
pipe = Pipeline(steps=[
    ('pre', preprocessing),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
```

---

## 🤖 3. Model Zoo — Best Use & Key Parameters
| Model Type | Classifiers | Best Use | Key Params |
|------------|-------------|----------|------------|
| Linear | LogisticRegression | Simple, interpretable | `C`, `penalty`, `class_weight` |
| Tree | DecisionTreeClassifier | Fast, interpretable | `max_depth`, `min_samples_split` |
| Ensemble | RandomForestClassifier | Good default | `n_estimators`, `max_features` |
| Boosting | XGBClassifier, LGBMClassifier | Best accuracy | `learning_rate`, `n_estimators` |
| SVM | SVC | Small/medium datasets | `kernel`, `C`, `gamma` |
| kNN | KNeighborsClassifier | Simple logic | `n_neighbors` |
| Naive Bayes | GaussianNB | Independent features | — |
| Neural Nets | MLPClassifier | Non-linear patterns | `hidden_layer_sizes`, `alpha` |

---

## 🔄 4. Hyper-parameter Tuning
```python
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rnd = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42, n_jobs=-1, eval_metric='auc'),
    param_distributions={'max_depth':[3,5,7], 'learning_rate':[0.05,0.1,0.2]},
    n_iter=20,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)
rnd.fit(X_train, y_train)
```

---

## 📈 5. Evaluation Metrics & Plots
- `roc_auc_score`, `average_precision_score`
- `classification_report`, `confusion_matrix`
- ROC & Precision-Recall curve
```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
```
- Calibration curve: `CalibratedClassifierCV`

---

## 🔍 6. Interpretability
- Trees: `.feature_importances_`, permutation importance
- Boosting: SHAP (`shap.TreeExplainer`)
- Logistic: standardized coefficients
- Share business insights from important features

---

## 💾 7. Model Persistence
```python
import joblib
joblib.dump(best_model, 'cross_sell_model.joblib')
```

---

## 📚 References
- [scikit-learn Model Map](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)


## Checklist
 ### Explore the Data
 Note: try to get insights from a field expert for these steps.
 1. Create a copy of the data for exploration (sampling it down to a
 manageable size if necessary).
 2. Create a Jupyter notebook to keep a record of your data exploration.
 3. Study each attribute and its characteristics:
 
 - Name
 - Type (categorical, int/float, bounded/unbounded, text, structured,
 etc.)
 - % of missing values
 - Noisiness and type of noise (stochastic, outliers, rounding errors,
 etc.)
 - Usefulness for the task
 - Type of distribution (Gaussian, uniform, logarithmic, etc.)
 
 4. For supervised learning tasks, identify the target attribute(s).
 5. Visualize the data.
 6. Study the correlations between attributes.
 7. Study how you would solve the problem manually.
 8. Identify the promising transformations you may want to apply.
 9. Identify extra data that would be useful (go back to “Get the Data”).
10. Document what you have learned.

### Prepare the Data
 Notes:
 Work on copies of the data (keep the original dataset intact).
 Write functions for all data transformations you apply, for five reasons:
 
 - So you can easily prepare the data the next time you get a fresh
 dataset
 - So you can apply these transformations in future projects
 - To clean and prepare the test set
 - To clean and prepare new data instances once your solution is live
 - To make it easy to treat your preparation choices as
 hyperparameters
 
 1. Clean the data:
 Fix or remove outliers (optional).
 Fill in missing values (e.g., with zero, mean, median…) or drop
 their rows (or columns).
 2. Perform feature selection (optional):
 Drop the attributes that provide no useful information for the task.
 3. Perform feature engineering, where appropriate:
  - Discretize continuous features.
  - Decompose features (e.g., categorical, date/time, etc.).
 
  - Add promising transformations of features (e.g., log(x), sqrt(x),
 x , etc.).
  - Aggregate features into promising new features.
 4. Perform feature scaling:
  - Standardize or normalize features.
  - Shortlist Promising Models
 Notes:
 If the data is huge, you may want to sample smaller training sets so
 you can train many different models in a reasonable time (be aware
 that this penalizes complex models such as large neural nets or random
 forests).
 Once again, try to automate these steps as much as possible.
 1. Train many quick-and-dirty models from different categories
 (e.g., linear, naive Bayes, SVM, random forest, neural net, etc.) using
 standard parameters.
 2. Measure and compare their performance:
 For each model, use N-fold cross-validation and compute the
 mean and standard deviation of the performance measure on the
 N folds.
 3. Analyze the most significant variables for each algorithm.
 4. Analyze the types of errors the models make:
 What data would a human have used to avoid these errors?
 5. Perform a quick round of feature selection and engineering.
 6. Perform one or two more quick iterations of the five previous steps.
 7. Shortlist the top three to five most promising models, preferring
 models that make different types of errors.

 ### Fine-Tune the System
 Notes:
 You will want to use as much data as possible for this step, especially
 as you move toward the end of fine-tuning.
 As always, automate what you can.
 1. Fine-tune the hyperparameters using cross-validation:
 Treat your data transformation choices as hyperparameters,
 especially when you are not sure about them (e.g., if you’re not
 sure whether to replace missing values with zeros or with the
 median value, or to just drop the rows).
 Unless there are very few hyperparameter values to explore,
 prefer random search over grid search. If training is very long,
 you may prefer a Bayesian optimization approach (e.g., using
 Gaussian process priors, as described by Jasper Snoek et al. ).
 2. Try ensemble methods. Combining your best models will often
 produce better performance than running them individually.
 3. Once you are confident about your final model, measure its
 performance on the test set to estimate the generalization error.