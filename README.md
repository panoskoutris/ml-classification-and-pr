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

We tackle a large-scale multi-class classification problem using:
- **Training set**: `datasetTV.csv` with 8743 samples, 224 features, and labels in {1, 2, 3, 4, 5}
- **Test set**: `datasetTest.csv` with 6955 unlabeled samples

### Pipeline:
- Preprocessing (normalization, dimensionality reduction if needed)
- Model selection and evaluation using cross-validation
- Final training and inference
- Output predicted labels in `labels.npy` (1D array of length 6955)

The solution is designed for generalization and scalability. Models such as logistic regression, support vector machines, or ensemble classifiers may be employed.

---

## 🛠️ Resources Utilized

- Python 3
- Jupiter Notebook
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

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

✍️ Author
Panagiotis Koutris
Student at ECE AUTH – School of Electrical & Computer Engineering

📄 License
This project is licensed under the MIT License.




