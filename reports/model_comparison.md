# Model Comparison Summary

The following table summarizes the test-set performance of the trained classifiers.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9070 | 0.7706 | 0.9251 | 0.8408 | 0.9764 |
| Random Forest | 0.9212 | 0.7995 | 0.9385 | 0.8635 | 0.9686 |
| XGBoost | 0.9212 | 0.8329 | 0.8797 | 0.8557 | 0.9792 |
| CatBoost | 0.9198 | 0.7959 | 0.9385 | 0.8613 | 0.9818 |

## Best Model Selection
Random Forest was selected as the final model because it achieved the highest F1-score among the evaluated models on the test set, which is the most important metric for this imbalanced churn classification problem.
