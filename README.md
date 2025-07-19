# Pattern Recognition and Machine Learning Techniques

This repository explores a variety of supervised classification techniques, combining concepts from statistical pattern recognition and modern machine learning.

Developed as part of the course **Pattern Recognision and Machine Learning** at the Aristotle University of Thessaloniki (AUTH), School of Electrical & Computer Engineering.

The project is divided into four parts:
- **Parts A & B**: Analytical classification using probabilistic models
- **Part C**: Classical ML classifiers on low-dimensional data
- **Part D**: Scalable classification on high-dimensional, multi-class data

---

## 🧩 Part A – Lorentzian-Based Classification (Maximum Likelihood)

We construct a binary classifier for detecting user stress levels using a custom indicator derived from game controller input data. The data is assumed to follow a **Lorentzian (Cauchy)** distribution:

``p(x | θ) = 1 / π * 1 / (1 + (x − θ)^2)``

Tasks:
- Estimate the unknown parameter θ for each class using **Maximum Likelihood Estimation (MLE)**
- Visualize log-likelihoods across θ values
- Implement a discriminant function:

``g(x) = log p(x | θ1) − log p(x | θ2) + log P(ω1) − log P(ω2)``


- Analyze the decision boundary and classification accuracy on training points

---

## 🧩 Part B – Bayesian Parameter Estimation and Classification

Building on Part A, we now incorporate a **prior distribution** for θ:

``p(θ) = 1 / (10π) * 1 / (1 + (θ / 10)^2)``

Tasks:
- Compute the posterior `p(θ | D)` using numerical integration (trapezoidal rule)
- Estimate `p(x | D_j)` via marginalization
- Construct and visualize a new discriminant function:

``h(x) = log p(x | D1) − log p(x | D2) + log P(ω1) − log P(ω2)``


- Compare the Bayesian classifier to the MLE-based one

---

## 🧩 Part C – Decision Trees and Random Forests on Iris Data

We apply classical ML classifiers to a subset of the Iris dataset, using only:
- Sepal length
- Sepal width

### Subpart 1: Decision Tree
- Split the dataset into 50% training / 50% test
- Train a `DecisionTreeClassifier` using `scikit-learn`
- Evaluate accuracy and tune tree depth
- Visualize decision boundaries with `contourf`

### Subpart 2: Random Forest (Bootstrap Aggregation)
- Construct a random forest with 100 decision trees
- Each tree is trained on a bootstrap sample from the training set
- Evaluate accuracy on the held-out test set
- Analyze improvements over a single decision tree
- Discuss the impact of the sample ratio γ

---

## 🧩 Part D – Classification of High-Dimensional Data

This part focuses on solving a large-scale, multi-class classification problem using a real-world dataset:

- **Training set**: `datasetTV.csv` — 8743 samples, each with 224 features and a class label in {1, 2, 3, 4, 5}
- **Test set**: `datasetTest.csv` — 6955 unlabeled samples with the same 224 features

### Approach

The classification process included the following steps:

1. **Preprocessing and Dimensionality Reduction**
   - All features were standardized using `StandardScaler` to ensure they have zero mean and unit variance.
   - **Principal Component Analysis (PCA)** was applied to reduce the number of features while retaining the majority of the dataset's variance. The number of components was chosen based on the cumulative explained variance.

2. **Model Selection and Training**
   - Several classifiers were tested, and a **Random Forest** was selected due to its consistent performance and robustness.
   - Key hyperparameters such as the number of trees (`n_estimators`) and maximum depth were adjusted manually.
   - The training set was internally split to validate the model's performance before final deployment.

3. **Prediction**
   - After training, the final model was used to predict labels for the test set (`datasetTest.csv`).
   - The predictions were saved in a file named `labels.npy` as a 1D NumPy array of length 6955.

### Output

- `labels.npy` contains the predicted class labels for each test sample.
- All preprocessing and classification steps were implemented using standard Python libraries such as NumPy, scikit-learn, and Pandas.


---

## 🛠️ Resources Utilized

- Python 3
- Jupiter Notebook
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

---

## 📁 Repository Structure

```
ml-classification-and-pr/
├── README.md                 # Project overview, explanation of methods, and usage instructions
├── parts-a-to-c.ipynb        # Notebook for Parts A–C: MLE, Bayesian classification, decision trees, random forests
├── part-d.ipynb              # Notebook for Part D: high-dimensional multi-class classification
├── datasetTV.csv             # Training data for Part D (8743 samples × 224 features + labels)
├── datasetTest.csv           # Test data for Part D (6955 samples × 224 features, unlabeled)
├── labels.npy                # Predicted labels for datasetTest (1D NumPy array, output of part-d.ipynb)
└── report-presentation.pdf   # Slide-based summary of all parts: methodology, results, visualizations
```

---

## ✍️ Author

Panagiotis Koutris
Student at ECE AUTH – School of Electrical & Computer Engineering

---

## 📄 License

This project is licensed under the MIT License.




