# NFL Play Prediction using Machine Learning

This project uses pre-snap player movement and game context data from the [NFL Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data) to predict offensive play types (pass vs. run) using machine learning techniques. We developed and evaluated multiple models—including logistic regression, decision trees, random forest, and ensemble methods—to automate play calling insights.

---

## Dataset

We used three datasets from the 2025 Big Data Bowl:
- `player_play.csv`: Frame-by-frame player movement and metrics.
- `play.csv`: Contextual game information for each play.
- `player.csv`: Player biographical and physical data.

All datasets were cleaned and merged to retain only pre-snap features for realistic prediction.

---

## Models Used

We trained and evaluated the following models:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Ensemble (Soft Voting)** – weighted combination of all models

---

## Methodology

- **Feature Engineering**  
  - Created new features such as absolute field position, quarterback ID, and a binary `is_pass` target.  
  - Converted time to numerical format (seconds remaining).  
  - One-hot encoded categorical features and handled null values.

- **Dimensionality Analysis**  
  - PCA explored but not used due to loss of categorical information.

- **Model Training**  
  - 75-25 train/test split with 80-20 train/validation.  
  - Hyperparameter tuning on tree depth and number of estimators.  
  - 5-fold cross-validation to evaluate generalization.

---

## Results

| Model                  | Test Accuracy | Avg F1 Score |
|------------------------|---------------|--------------|
| Logistic Regression    | 72.4%         | 0.65         |
| Decision Tree          | 72.8%         | 0.66         |
| Random Forest          | **73.4%**     | **0.67**     |
| Ensemble (Soft Voting) | 73.1%         | 0.67         |

> **Random Forest** yielded the highest test accuracy and generalization performance.

---

## Ablation Study

- Removing key features like `expectedPoints` and `fieldPosition` decreased accuracy by 2–5%.
- Untuned models often overfit (e.g., 100% training accuracy with low test accuracy).
- Ensemble weighting improved performance by leveraging the strengths of individual models.

---

## Key Takeaways

- Tree-based models outperform logistic regression in predicting NFL play types from pre-snap data.
- Ensemble methods provide a performance boost by combining strengths of individual classifiers.
- Data-driven strategies can supplement traditional film study in football analytics.

---


## References

- [NFL Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data)  
- [Goyal, U. (2019). MIT – ML for Play Calling](https://dspace.mit.edu/handle/1721.1/129909)  
- [Teich et al., 2016. NFL Play Prediction](https://arxiv.org/abs/1601.00574)

---

## Future Work

- Incorporate deep learning models like neural networks  
- Predict additional targets (e.g., formation type, play direction)  
- Integrate external context: weather, crowd noise, or coaching tendencies
