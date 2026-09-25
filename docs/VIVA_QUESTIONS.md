# Viva Questions & Answers
## Machine Learning Based Burning Feet Syndrome Symptom Assessment

1. **What is Burning Feet Syndrome?**
   A general term for a group of foot sensations — burning, tingling, numbness, pain — often
   discussed in connection with diabetes, vitamin deficiency, thyroid issues, alcohol use, or
   peripheral neuropathy.

2. **Why did you select this topic?**
   It offers a realistic, structured (tabular) multi-class classification problem with clear
   symptom/health-history features, suitable for demonstrating a full applied ML workflow.

3. **What is machine learning?**
   A field of computer science where systems learn patterns from data and make predictions or
   decisions without being explicitly programmed with fixed rules.

4. **What type of ML problem is this?**
   Supervised multi-class classification.

5. **Why is this classification and not regression?**
   Because the target variable (`assessment`) is a discrete category (Low/Moderate/High), not a
   continuous numeric value.

6. **What is supervised learning?**
   Learning a mapping from input features to a known output label using a labeled training
   dataset.

7. **What is the target variable in this project?**
   `assessment`, with three classes: Low, Moderate, High.

8. **What are features?**
   The input variables used to make a prediction — e.g., age, symptoms, duration, health history.

9. **What is preprocessing, and why is it needed?**
   Transforming raw data (handling missing values, encoding categories, scaling numbers) into a
   clean, numeric format models can learn from effectively.

10. **Why do we handle missing values?**
    Most ML algorithms cannot process `NaN` values directly, and missing data can bias or break
    training if left unhandled.

11. **How did you handle missing values here?**
    Numeric columns were imputed with the median; categorical columns with the mode (most
    frequent value).

12. **Why is feature scaling required?**
    Algorithms like SVM, KNN and Logistic Regression are sensitive to feature magnitude;
    unscaled features with large ranges can dominate the learned decision boundary.

13. **Which scaler did you use, and why?**
    `StandardScaler`, which standardizes features to zero mean and unit variance — a common,
    robust default for numeric features.

14. **What is train-test split?**
    Dividing the dataset into a training set (to fit the model) and a test set (to evaluate it
    on unseen data), to estimate real-world generalization.

15. **What split ratio did you use?**
    80% training, 20% testing, with `stratify=y` to preserve class proportions.

16. **Why use `stratify=y` in the split?**
    To ensure each class (Low/Moderate/High) is represented in the same proportion in both the
    training and test sets, which matters for imbalanced or moderately imbalanced data.

17. **What is Logistic Regression?**
    A linear model that estimates class probabilities using a logistic (softmax, for multi-class)
    function of a weighted sum of input features.

18. **What is a Decision Tree?**
    A model that splits data into branches based on feature thresholds, forming a tree of
    if-else rules, ending in leaf nodes that predict a class.

19. **What is Random Forest?**
    An ensemble of many decision trees, each trained on a random subset of data/features
    (bagging); predictions are aggregated (majority vote) to reduce overfitting and variance.

20. **What is KNN?**
    K-Nearest Neighbors — a non-parametric method that classifies a new point based on the
    majority class among its `k` closest points in the training data.

21. **What is SVM?**
    Support Vector Machine — a classifier that finds the optimal boundary (hyperplane) that
    maximizes the margin between classes; here using an RBF kernel to handle non-linear
    boundaries.

22. **Why did you compare multiple algorithms instead of picking one?**
    Different algorithms have different strengths/assumptions; comparing them empirically
    ensures the final choice is justified by actual performance rather than assumption.

23. **Which model performed best, and why?**
    In this run, a tuned SVM (and closely, Logistic Regression) achieved the highest macro
    F1-score (~0.87) on the held-out test set.

24. **What is overfitting?**
    When a model learns the training data too closely (including noise), performing well on
    training data but poorly on new, unseen data.

25. **What is underfitting?**
    When a model is too simple to capture the underlying pattern in the data, performing poorly
    on both training and test data.

26. **How did you guard against overfitting?**
    Using cross-validation to check consistency across folds, comparing CV performance to test
    performance, and preferring simpler/tuned models over unnecessarily complex ones.

27. **What is cross-validation?**
    A technique that splits the training data into multiple folds, training/evaluating the model
    several times on different fold combinations to get a more reliable performance estimate.

28. **What cross-validation strategy did you use?**
    Stratified 5-fold cross-validation, which preserves class proportions in each fold.

29. **What is GridSearchCV?**
    A hyperparameter-tuning technique that exhaustively tries every combination of specified
    parameter values, using cross-validation to score each, and returns the best-performing
    combination.

30. **Which hyperparameters did you tune, and why?**
    For SVM: `C` (regularization strength), `kernel` (decision boundary shape), and `gamma`
    (kernel coefficient) — these are the parameters with the largest impact on SVM's bias-variance
    tradeoff and non-linear flexibility.

31. **What is accuracy?**
    The proportion of total predictions that were correct.

32. **What is precision?**
    Of all instances predicted as a given class, the proportion that were actually that class
    (measures false-positive control).

33. **What is recall?**
    Of all instances that actually belong to a given class, the proportion the model correctly
    identified (measures false-negative control).

34. **What is F1-score?**
    The harmonic mean of precision and recall, balancing both; useful when classes matter
    similarly and/or are imbalanced.

35. **Why did you use macro-averaging for precision/recall/F1?**
    Macro-averaging computes the metric independently per class and averages them, giving equal
    importance to each class regardless of class size — appropriate here since Low/Moderate/High
    are not equally frequent but are equally important to predict correctly.

36. **What is a confusion matrix?**
    A table showing counts of actual vs. predicted classes, revealing exactly which classes are
    being confused with which.

37. **What is feature importance?**
    A measure of how much each input feature contributes to a model's predictions.

38. **How did you compute feature importance for SVM, which has no native importance scores?**
    Using **permutation importance** — shuffling one feature's values at a time and measuring
    the resulting drop in model performance (F1-macro) on the test set; a larger drop means
    higher importance.

39. **Why use Streamlit for the interface?**
    Streamlit allows building an interactive Python-only web app quickly, without needing
    separate frontend/backend frameworks — well suited to demonstrating an ML pipeline.

40. **How is the trained model saved and reused?**
    Using `joblib.dump()` to serialize the entire fitted pipeline (preprocessing + classifier) to
    `models/burning_feet_model.pkl`, then `joblib.load()` in the Streamlit app / prediction
    script to reuse it without retraining.

41. **What is Joblib, and why use it instead of `pickle`?**
    A serialization library optimized for objects with large NumPy arrays (common in
    scikit-learn models); it's the standard recommended way to persist scikit-learn pipelines.

42. **What are the limitations of your project?**
    The dataset is entirely synthetic (no real clinical validation), the feature set is
    simplified, and class boundaries come from an arbitrary synthetic scoring rule rather than
    medical consensus.

43. **Why can't your model be considered a medical diagnosis?**
    It was trained only on synthetic data with no real-world clinical outcomes behind it, so its
    predictions reflect patterns in generated data, not validated medical relationships.

44. **How can this project be improved in the future?**
    By incorporating properly consented real-world data (with clinical oversight), validated
    symptom-assessment questionnaires, additional model types (e.g., gradient boosting), and a
    deployed API layer.

45. **Why did you use a `Pipeline`/`ColumnTransformer` instead of manually preprocessing the
    data?**
    It bundles preprocessing and modeling into one object, ensures preprocessing statistics
    (like scaler mean/std) are fit only on training data (preventing data leakage), and makes
    the whole system trivially reusable in the Streamlit app.
