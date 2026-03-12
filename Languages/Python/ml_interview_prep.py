"""
============================================================================
MACHINE LEARNING INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
For 8+ Years SDE Experience
============================================================================

Topics Covered:
1. Supervised Learning (Regression, Classification)
2. Unsupervised Learning (Clustering, Dimensionality Reduction)
3. Deep Learning (Neural Networks, CNN, RNN, Transformers)
4. Feature Engineering & Preprocessing
5. Model Evaluation & Validation
6. Hyperparameter Tuning
7. Ensemble Methods
8. Time Series Analysis
9. Natural Language Processing
10. Computer Vision
11. Model Deployment & MLOps
12. Common ML Algorithms from Scratch
13. Advanced Topics (Transfer Learning, GANs, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LINEAR REGRESSION - FROM SCRATCH AND SKLEARN
# ============================================================================
# WHEN TO USE:
# - Predicting continuous values (e.g., house prices, sales).
# - Relationship between variables is linear.
# - Need interpretability (coefficients show feature impact).
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Non-linear data):
# - Feature engineering: Add polynomial features or interaction terms.
# - Log-transform target variable if skewed.
# - Use regularization (Ridge/Lasso) if overfitting.
# ============================================================================

class LinearRegressionFromScratch:
    """
    Linear Regression using Gradient Descent
    y = mx + b (simple) or y = β₀ + β₁x₁ + β₂x₂ + ... (multiple)
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y - y_predicted) ** 2)
            self.losses.append(loss)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
def linear_regression_example():
    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Train model
    model = LinearRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y.ravel())
    
    # Make predictions
    predictions = model.predict(X)
    
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print(f"Final Loss: {model.losses[-1]:.4f}")

# Using sklearn
def sklearn_linear_regression():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Generate data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

# Ridge Regression (L2 Regularization)
def ridge_regression_example():
    from sklearn.linear_model import Ridge
    
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    # alpha is the regularization parameter
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    print(f"Ridge coefficients: {model.coef_}")

# Lasso Regression (L1 Regularization)
def lasso_regression_example():
    from sklearn.linear_model import Lasso
    
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    # L1 regularization can zero out coefficients (feature selection)
    model = Lasso(alpha=0.1)
    model.fit(X, y)
    
    print(f"Lasso coefficients: {model.coef_}")
    print(f"Non-zero features: {np.sum(model.coef_ != 0)}")

# ============================================================================
# 2. LOGISTIC REGRESSION - BINARY CLASSIFICATION
# ============================================================================
# WHEN TO USE:
# - Binary classification problems (e.g., spam vs ham, churn vs stay).
# - Need probability outputs.
# - Linearly separable classes.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Non-linear boundaries):
# - Feature engineering: Add polynomial features.
# - Use Kernel methods or switch to non-linear models like Random Forest/SVM.
# - Handle class imbalance with class weights or resampling.
# ============================================================================

class LogisticRegressionFromScratch:
    """
    Binary Classification using Sigmoid function
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Apply sigmoid
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted > 0.5).astype(int)

# Using sklearn
def sklearn_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# ============================================================================
# 3. DECISION TREES AND RANDOM FORESTS
# ============================================================================
# WHEN TO USE:
# - Tabular data with mixed feature types (numerical/categorical).
# - Non-linear relationships.
# - Need interpretability (Decision Trees).
# - High accuracy and robustness (Random Forests).
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Overfitting):
# - Pruning: Limit max_depth, min_samples_split.
# - Ensembling: Use Random Forest or Gradient Boosting.
# - Feature selection to remove noise.
# ============================================================================

def decision_tree_example():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train decision tree
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion='gini'  # or 'entropy'
    )
    dt.fit(X_train, y_train)
    
    # Predict
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    print(f"Feature Importances: {dt.feature_importances_}")
    
    return dt

def random_forest_example():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,      # number of trees
        max_depth=10,
        min_samples_split=2,
        max_features='sqrt',   # sqrt(n_features) for each split
        random_state=42,
        n_jobs=-1             # use all CPUs
    )
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print(f"Top 5 Feature Importances: {np.argsort(rf.feature_importances_)[-5:]}")
    
    return rf

# Gradient Boosting
def gradient_boosting_example():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, gb.predict(X_test))
    print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
    
    return gb

# XGBoost (if available)
def xgboost_example():
    try:
        import xgboost as xgb
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # XGBoost Classifier
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        
        return model
    except ImportError:
        print("XGBoost not installed")

# ============================================================================
# 4. SUPPORT VECTOR MACHINES (SVM)
# ============================================================================
# WHEN TO USE:
# - Small to medium sized datasets.
# - High dimensional spaces (e.g., text classification).
# - Clear margin of separation.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Slow training):
# - Use LinearSVC for large datasets (faster).
# - Scale data (critical for SVM).
# - Reduce dimensionality (PCA) before training.
# ============================================================================

def svm_example():
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    
    # Generate data
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Linear SVM
    svm_linear = SVC(kernel='linear', C=1.0)
    svm_linear.fit(X_train, y_train)
    
    # RBF (Radial Basis Function) kernel
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_rbf.fit(X_train, y_train)
    
    # Polynomial kernel
    svm_poly = SVC(kernel='poly', degree=3, C=1.0)
    svm_poly.fit(X_train, y_train)
    
    # Evaluate
    print(f"Linear SVM Accuracy: {svm_linear.score(X_test, y_test):.4f}")
    print(f"RBF SVM Accuracy: {svm_rbf.score(X_test, y_test):.4f}")
    print(f"Poly SVM Accuracy: {svm_poly.score(X_test, y_test):.4f}")
    
    return svm_rbf

# ============================================================================
# 5. K-NEAREST NEIGHBORS (KNN)
# ============================================================================
# WHEN TO USE:
# - Small datasets.
# - Simple baseline.
# - Instance-based learning (no training phase).
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Slow prediction):
# - Use KD-Tree or Ball-Tree algorithms.
# - Reduce dimensionality.
# - Use approximate nearest neighbors (ANN) for very large datasets.
# ============================================================================

class KNNFromScratch:
    """K-Nearest Neighbors Classifier"""
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute distances
        distances = [self.euclidean_distance(x, x_train) 
                    for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority vote
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Using sklearn
def sklearn_knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train KNN
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',  # or 'distance'
        algorithm='auto',   # 'ball_tree', 'kd_tree', 'brute'
        metric='euclidean'  # or 'manhattan', 'minkowski'
    )
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"KNN Accuracy: {accuracy:.4f}")
    
    return knn

# ============================================================================
# 6. NAIVE BAYES
# ============================================================================
# WHEN TO USE:
# - Text classification (spam filtering, sentiment analysis).
# - High dimensional data.
# - When independence assumption holds (reasonably).
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Zero frequency):
# - Use Laplace smoothing (alpha parameter).
# - Log probabilities to prevent underflow.
# ============================================================================

def naive_bayes_example():
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Gaussian Naive Bayes (for continuous features)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, gnb.predict(X_test))
    print(f"Gaussian Naive Bayes Accuracy: {accuracy:.4f}")
    
    return gnb

# ============================================================================
# 7. CLUSTERING - UNSUPERVISED LEARNING
# ============================================================================
# WHEN TO USE:
# - Discovering structure in unlabeled data.
# - Customer segmentation.
# - Anomaly detection.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., K-Means assumes spherical clusters):
# - Use DBSCAN for arbitrary shapes.
# - Use Gaussian Mixture Models for soft clustering.
# - Scale data before clustering.
# ============================================================================

def kmeans_example():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    
    # Generate data
    X, y_true = make_blobs(
        n_samples=300, centers=4, n_features=2,
        cluster_std=0.60, random_state=42
    )
    
    # Train K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Predict cluster labels
    y_pred = kmeans.predict(X)
    
    print(f"Cluster centers:\n{kmeans.cluster_centers_}")
    print(f"Inertia: {kmeans.inertia_:.4f}")
    
    return kmeans

# K-Means from scratch
class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
    
    def fit(self, X):
        # Initialize centroids randomly
        random_indices = np.random.choice(
            len(X), self.n_clusters, replace=False
        )
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            # Assign clusters
            clusters = self._assign_clusters(X)
            
            # Update centroids
            old_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X, clusters)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
    
    def _assign_clusters(self, X):
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, clusters):
        return np.array([
            X[clusters == i].mean(axis=0)
            for i in range(self.n_clusters)
        ])
    
    def predict(self, X):
        return self._assign_clusters(X)

# Hierarchical Clustering
def hierarchical_clustering_example():
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    
    # Hierarchical clustering
    hc = AgglomerativeClustering(
        n_clusters=4,
        linkage='ward'  # 'complete', 'average', 'single'
    )
    y_pred = hc.fit_predict(X)
    
    print(f"Cluster labels: {np.unique(y_pred)}")
    
    return hc

# DBSCAN (Density-Based Clustering)
def dbscan_example():
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons
    
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_pred = dbscan.fit_predict(X)
    
    print(f"Number of clusters: {len(np.unique(y_pred)) - 1}")  # -1 for noise
    print(f"Number of noise points: {np.sum(y_pred == -1)}")
    
    return dbscan

# ============================================================================
# 8. DIMENSIONALITY REDUCTION
# ============================================================================
# WHEN TO USE:
# - Visualize high-dimensional data (2D/3D).
# - Reduce noise and computational cost.
# - Remove multicollinearity.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Information loss):
# - Check explained variance ratio (keep >95%).
# - Use t-SNE or UMAP for visualization (preserves local structure).
# - Use Autoencoders for non-linear reduction.
# ============================================================================

def pca_example():
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_digits
    
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions
    X_pca = pca.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Determine optimal number of components
    pca_full = PCA()
    pca_full.fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= 0.95) + 1  # 95% variance
    print(f"Components for 95% variance: {n_components}")
    
    return pca

# t-SNE for visualization
def tsne_example():
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )
    X_tsne = tsne.fit_transform(X)
    
    print(f"t-SNE shape: {X_tsne.shape}")
    
    return X_tsne

# LDA (Linear Discriminant Analysis)
def lda_example():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # LDA (supervised dimensionality reduction)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    
    print(f"LDA shape: {X_lda.shape}")
    print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
    
    return lda

# ============================================================================
# 9. NEURAL NETWORKS - BASICS
# ============================================================================
# WHEN TO USE:
# - Complex patterns (images, audio, text).
# - Large amount of data available.
# - High accuracy required.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Vanishing gradients):
# - Use ReLU activation instead of Sigmoid/Tanh.
# - Use Batch Normalization.
# - Use residual connections (ResNet).
# ============================================================================

class SimpleNeuralNetwork:
    """
    Simple feedforward neural network with one hidden layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output, learning_rate=0.01):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = np.mean((y - output) ** 2)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        return self.forward(X)

# Using sklearn MLPClassifier
def sklearn_neural_network():
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Multi-layer Perceptron
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 hidden layers
        activation='relu',              # 'relu', 'tanh', 'logistic'
        solver='adam',                  # 'sgd', 'adam'
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, mlp.predict(X_test))
    print(f"MLP Accuracy: {accuracy:.4f}")
    print(f"Number of layers: {mlp.n_layers_}")
    print(f"Number of iterations: {mlp.n_iter_}")
    
    return mlp

# ============================================================================
# 10. DEEP LEARNING WITH KERAS/TENSORFLOW
# ============================================================================
# WHEN TO USE:
# - State-of-the-art performance on unstructured data.
# - End-to-end learning.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Training instability):
# - Learning rate scheduling.
# - Early stopping.
# - Gradient clipping.
# ============================================================================

def build_simple_neural_network():
    """Simple feedforward network with Keras"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(20,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except ImportError:
        print("TensorFlow not installed")
        return None

def build_cnn_for_images():
    """Convolutional Neural Network for image classification"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except ImportError:
        print("TensorFlow not installed")
        return None

def build_rnn_for_sequences():
    """Recurrent Neural Network for sequence data"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(None, 10)),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except ImportError:
        print("TensorFlow not installed")
        return None

# ============================================================================
# 11. FEATURE ENGINEERING AND PREPROCESSING
# ============================================================================
# WHEN TO USE:
# - Always! "Garbage in, garbage out".
# - To improve model performance and convergence.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Data leakage):
# - Use sklearn Pipelines to encapsulate preprocessing.
# - Perform split BEFORE any preprocessing.
# ============================================================================

def feature_scaling_examples():
    """Different scaling techniques"""
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, Normalizer
    )
    
    X = np.random.randn(100, 5) * 10 + 50
    
    # StandardScaler: (x - mean) / std
    scaler_standard = StandardScaler()
    X_standard = scaler_standard.fit_transform(X)
    print(f"Standard scaled mean: {X_standard.mean(axis=0)}")
    print(f"Standard scaled std: {X_standard.std(axis=0)}")
    
    # MinMaxScaler: (x - min) / (max - min)
    scaler_minmax = MinMaxScaler()
    X_minmax = scaler_minmax.fit_transform(X)
    print(f"MinMax scaled range: [{X_minmax.min()}, {X_minmax.max()}]")
    
    # RobustScaler: uses median and IQR (robust to outliers)
    scaler_robust = RobustScaler()
    X_robust = scaler_robust.fit_transform(X)
    
    # Normalizer: scales each sample to unit norm
    normalizer = Normalizer()
    X_normalized = normalizer.fit_transform(X)
    
    return X_standard, X_minmax, X_robust, X_normalized

def encoding_categorical_features():
    """Encoding categorical variables"""
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # Label Encoding (for ordinal features)
    labels = ['low', 'medium', 'high', 'medium', 'low']
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    print(f"Label encoded: {labels_encoded}")
    
    # One-Hot Encoding (for nominal features)
    categories = np.array(['cat', 'dog', 'bird', 'cat', 'dog']).reshape(-1, 1)
    ohe = OneHotEncoder(sparse_output=False)
    categories_encoded = ohe.fit_transform(categories)
    print(f"One-hot encoded shape: {categories_encoded.shape}")
    
    # Using pandas get_dummies
    df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
    df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
    print(df_encoded)
    
    return labels_encoded, categories_encoded

def handle_missing_values():
    """Different strategies for handling missing data"""
    from sklearn.impute import SimpleImputer
    
    # Create data with missing values
    X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, np.nan]])
    
    # Mean imputation
    imputer_mean = SimpleImputer(strategy='mean')
    X_mean = imputer_mean.fit_transform(X)
    
    # Median imputation
    imputer_median = SimpleImputer(strategy='median')
    X_median = imputer_median.fit_transform(X)
    
    # Most frequent imputation
    imputer_frequent = SimpleImputer(strategy='most_frequent')
    X_frequent = imputer_frequent.fit_transform(X)
    
    # Constant imputation
    imputer_constant = SimpleImputer(strategy='constant', fill_value=0)
    X_constant = imputer_constant.fit_transform(X)
    
    print(f"Original:\n{X}")
    print(f"Mean imputed:\n{X_mean}")
    
    return X_mean

def feature_selection_examples():
    """Feature selection techniques"""
    from sklearn.feature_selection import (
        SelectKBest, f_classif, RFE, SelectFromModel
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Lasso
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=10, random_state=42)
    
    # 1. Univariate Selection (SelectKBest)
    selector_kbest = SelectKBest(f_classif, k=10)
    X_kbest = selector_kbest.fit_transform(X, y)
    print(f"SelectKBest selected features: {selector_kbest.get_support()}")
    
    # 2. Recursive Feature Elimination (RFE)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_rfe = RFE(rf, n_features_to_select=10)
    X_rfe = selector_rfe.fit_transform(X, y)
    print(f"RFE selected features: {selector_rfe.support_}")
    
    # 3. L1-based feature selection (Lasso)
    lasso = Lasso(alpha=0.05, random_state=42)
    selector_l1 = SelectFromModel(lasso)
    selector_l1.fit(X, y)
    X_l1 = selector_l1.transform(X)
    print(f"L1 selected features: {selector_l1.get_support()}")
    
    # 4. Tree-based feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    top_features = np.argsort(feature_importances)[-10:]
    print(f"Top 10 features by importance: {top_features}")
    
    return X_kbest

def polynomial_features_example():
    """Creating polynomial features"""
    from sklearn.preprocessing import PolynomialFeatures
    
    X = np.array([[1, 2], [3, 4]])
    
    # Degree 2 polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Polynomial shape: {X_poly.shape}")
    print(f"Feature names: {poly.get_feature_names_out()}")
    
    return X_poly

# ============================================================================
# 12. MODEL EVALUATION AND VALIDATION
# ============================================================================
# WHEN TO USE:
# - To assess model performance on unseen data.
# - To compare different models.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Metric mismatch):
# - Use custom scoring functions.
# - Look at multiple metrics (Precision/Recall trade-off).
# - Use domain-specific business metrics (e.g., profit/loss).
# ============================================================================

def cross_validation_example():
    """Different cross-validation strategies"""
    from sklearn.model_selection import (
        KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
    )
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"K-Fold CV scores: {scores_kfold}")
    print(f"Mean: {scores_kfold.mean():.4f} (+/- {scores_kfold.std():.4f})")
    
    # Stratified K-Fold (preserves class distribution)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_skfold = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
    print(f"Stratified K-Fold CV scores: {scores_skfold}")
    
    # Time Series Split (for temporal data)
    tscv = TimeSeriesSplit(n_splits=5)
    scores_ts = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    print(f"Time Series CV scores: {scores_ts}")
    
    return scores_kfold

def classification_metrics():
    """All important classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, precision_recall_curve
    )
    
    # Example predictions
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.3, 0.6, 0.85, 0.15])
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    
    return accuracy, precision, recall, f1

def regression_metrics():
    """Regression evaluation metrics"""
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error
    )
    
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    
    return mse, rmse, mae, r2

def learning_curves():
    """Plotting learning curves to detect overfitting/underfitting"""
    from sklearn.model_selection import learning_curve
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    print(f"Training sizes: {train_sizes}")
    print(f"Training scores: {train_mean}")
    print(f"Validation scores: {val_mean}")
    
    return train_sizes, train_mean, val_mean

# ============================================================================
# 13. HYPERPARAMETER TUNING
# ============================================================================
# WHEN TO USE:
# - To optimize model performance.
# - When default parameters are not sufficient.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Too expensive):
# - Use Random Search or Bayesian Optimization instead of Grid Search.
# - Tune on a subset of data.
# - Use HalvingGridSearchCV for faster pruning.
# ============================================================================

def grid_search_example():
    """Grid Search for hyperparameter tuning"""
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid Search
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def random_search_example():
    """Random Search for hyperparameter tuning"""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint, uniform
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Define parameter distributions
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # Random Search
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=50, cv=5,
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def bayesian_optimization_example():
    """Bayesian optimization (using optuna if available)"""
    try:
        import optuna
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            
            model = RandomForestClassifier(**params, random_state=42)
            score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    except ImportError:
        print("Optuna not installed")

# ============================================================================
# 14. ENSEMBLE METHODS
# ============================================================================
# WHEN TO USE:
# - To improve accuracy and robustness over single models.
# - Reduce variance (Bagging) or bias (Boosting).
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Complexity/Latency):
# - Use model distillation (train a smaller student model).
# - Use simpler ensembles (Voting) instead of Stacking.
# ============================================================================

def voting_classifier_example():
    """Voting ensemble of multiple classifiers"""
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Individual models
    lr = LogisticRegression(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    svc = SVC(probability=True, random_state=42)
    
    # Hard voting (majority vote)
    voting_hard = VotingClassifier(
        estimators=[('lr', lr), ('dt', dt), ('svc', svc)],
        voting='hard'
    )
    voting_hard.fit(X_train, y_train)
    score_hard = voting_hard.score(X_test, y_test)
    
    # Soft voting (average probabilities)
    voting_soft = VotingClassifier(
        estimators=[('lr', lr), ('dt', dt), ('svc', svc)],
        voting='soft'
    )
    voting_soft.fit(X_train, y_train)
    score_soft = voting_soft.score(X_test, y_test)
    
    print(f"Hard Voting Accuracy: {score_hard:.4f}")
    print(f"Soft Voting Accuracy: {score_soft:.4f}")
    
    return voting_soft

def bagging_example():
    """Bagging (Bootstrap Aggregating)"""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Base estimator
    base_estimator = DecisionTreeClassifier(random_state=42)
    
    # Bagging
    bagging = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    
    accuracy = bagging.score(X_test, y_test)
    print(f"Bagging Accuracy: {accuracy:.4f}")
    
    return bagging

def stacking_example():
    """Stacking ensemble"""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Base estimators
    estimators = [
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier())
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression()
    
    # Stacking
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5
    )
    stacking.fit(X_train, y_train)
    
    accuracy = stacking.score(X_test, y_test)
    print(f"Stacking Accuracy: {accuracy:.4f}")
    
    return stacking

def adaboost_example():
    """AdaBoost (Adaptive Boosting)"""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Base estimator
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    
    # AdaBoost
    adaboost = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    adaboost.fit(X_train, y_train)
    
    accuracy = adaboost.score(X_test, y_test)
    print(f"AdaBoost Accuracy: {accuracy:.4f}")
    
    return adaboost

# ============================================================================
# 15. TIME SERIES ANALYSIS
# ============================================================================
# WHEN TO USE:
# - Data has temporal dependency.
# - Forecasting future values.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Non-stationarity):
# - Differencing (d parameter in ARIMA).
# - Log transformation.
# - Use LSTM or Prophet which handle some non-stationarity better.
# ============================================================================

def time_series_decomposition():
    """Decompose time series into trend, seasonal, and residual"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Create sample time series
    np.random.seed(42)
    date_range = pd.date_range(start='2020-01-01', periods=365, freq='D')
    trend = np.linspace(0, 10, 365)
    seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, 365))
    noise = np.random.randn(365)
    ts = trend + seasonal + noise
    
    ts_series = pd.Series(ts, index=date_range)
    
    # Decompose
    decomposition = seasonal_decompose(ts_series, model='additive', period=30)
    
    trend_component = decomposition.trend
    seasonal_component = decomposition.seasonal
    residual_component = decomposition.resid
    
    print("Time series decomposed into trend, seasonal, and residual")
    
    return decomposition

def arima_model_example():
    """ARIMA model for time series forecasting"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Create sample time series
        np.random.seed(42)
        data = np.cumsum(np.random.randn(200))
        
        # Fit ARIMA model
        # ARIMA(p, d, q) where:
        # p: order of autoregressive part
        # d: degree of differencing
        # q: order of moving average part
        model = ARIMA(data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=10)
        
        print(f"ARIMA Summary:\n{fitted_model.summary()}")
        print(f"Forecast: {forecast}")
        
        return fitted_model
    except ImportError:
        print("statsmodels not installed")

def prophet_example():
    """Facebook Prophet for time series forecasting"""
    try:
        from prophet import Prophet
        
        # Create sample data
        date_range = pd.date_range(start='2020-01-01', periods=365, freq='D')
        y = np.cumsum(np.random.randn(365)) + 100
        
        df = pd.DataFrame({
            'ds': date_range,
            'y': y
        })
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        print(f"Forecast columns: {forecast.columns.tolist()}")
        
        return model, forecast
    except ImportError:
        print("Prophet not installed")

def lstm_for_time_series():
    """LSTM for time series forecasting"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        # Sample data
        data = np.sin(np.linspace(0, 100, 1000))
        seq_length = 10
        
        X, y = create_sequences(data, seq_length)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        print("LSTM model trained for time series")
        
        return model
    except ImportError:
        print("TensorFlow not installed")

# ============================================================================
# 16. NATURAL LANGUAGE PROCESSING
# ============================================================================
# WHEN TO USE:
# - Text data analysis.
# - Sentiment analysis, classification, translation.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., OOV words):
# - Use subword tokenization (BPE, WordPiece).
# - Use pretrained embeddings (GloVe, FastText) or Transformers.
# ============================================================================

def text_preprocessing():
    """Common text preprocessing steps"""
    import re
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    
    texts = [
        "Machine learning is awesome!",
        "Natural language processing is a subset of AI.",
        "Deep learning models are powerful."
    ]
    
    # Lowercase and remove punctuation
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Bag of Words (Count Vectorizer)
    count_vec = CountVectorizer()
    X_counts = count_vec.fit_transform(cleaned_texts)
    print(f"Vocabulary: {count_vec.get_feature_names_out()}")
    print(f"Count matrix shape: {X_counts.shape}")
    
    # TF-IDF (Term Frequency-Inverse Document Frequency)
    tfidf_vec = TfidfVectorizer()
    X_tfidf = tfidf_vec.fit_transform(cleaned_texts)
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    
    return X_counts, X_tfidf

def word_embeddings_example():
    """Word embeddings with Word2Vec"""
    try:
        from gensim.models import Word2Vec
        
        # Sample sentences
        sentences = [
            ['machine', 'learning', 'is', 'awesome'],
            ['natural', 'language', 'processing'],
            ['deep', 'learning', 'models']
        ]
        
        # Train Word2Vec
        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4
        )
        
        # Get word vector
        vector = model.wv['learning']
        print(f"Vector for 'learning': {vector[:5]}...")  # First 5 dims
        
        # Find similar words
        similar = model.wv.most_similar('learning', topn=3)
        print(f"Similar to 'learning': {similar}")
        
        return model
    except ImportError:
        print("Gensim not installed")

def sentiment_analysis_example():
    """Sentiment analysis with sklearn"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    # Sample data
    texts = [
        "I love this product!",
        "This is terrible.",
        "Great experience, highly recommend.",
        "Worst purchase ever.",
        "Amazing quality and service."
    ]
    labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative
    
    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])
    
    # Train
    pipeline.fit(texts, labels)
    
    # Predict
    test_texts = ["This is awesome!", "I hate it."]
    predictions = pipeline.predict(test_texts)
    
    print(f"Predictions: {predictions}")
    
    return pipeline

def text_classification_with_bert():
    """Text classification using pretrained BERT"""
    try:
        from transformers import BertTokenizer, TFBertForSequenceClassification
        import tensorflow as tf
        
        # Load pretrained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2
        )
        
        # Tokenize text
        texts = ["I love this!", "This is terrible."]
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            return_tensors='tf', max_length=128
        )
        
        # Get predictions
        outputs = model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        
        print(f"Predictions: {predictions}")
        
        return model
    except ImportError:
        print("Transformers library not installed")

# ============================================================================
# 17. COMPUTER VISION
# ============================================================================
# WHEN TO USE:
# - Image classification, object detection, segmentation.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Limited data):
# - Data Augmentation (rotate, flip, zoom).
# - Transfer Learning (use ImageNet pretrained models).
# ============================================================================

def image_preprocessing():
    """Common image preprocessing techniques"""
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Image Data Augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,              # Normalize pixels
            rotation_range=20,           # Random rotation
            width_shift_range=0.2,       # Horizontal shift
            height_shift_range=0.2,      # Vertical shift
            shear_range=0.2,            # Shear transformation
            zoom_range=0.2,             # Random zoom
            horizontal_flip=True,        # Random horizontal flip
            fill_mode='nearest'
        )
        
        print("Image data augmentation configured")
        
        return datagen
    except ImportError:
        print("TensorFlow not installed")

def transfer_learning_example():
    """Transfer learning with pretrained model"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.applications import VGG16
        
        # Load pretrained model (without top classification layer)
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')  # 10 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with {len(model.layers)} layers")
        print(f"Trainable parameters: {model.count_params()}")
        
        return model
    except ImportError:
        print("TensorFlow not installed")

def cnn_concept():
    """CNN Architecture Concepts"""
    concepts = {
        'Convolution': 'Extracts features using filters/kernels',
        'Pooling': 'Downsamples feature maps (Max, Average)',
        'Stride': 'Step size of the filter',
        'Padding': 'Adding border pixels to preserve spatial dimensions',
        'Flattening': 'Converting 2D/3D feature maps to 1D vector'
    }
    
    print("CNN Concepts:")
    for concept, desc in concepts.items():
        print(f"{concept}: {desc}")
    
    return concepts

def object_detection_concept():
    """Object detection concepts (YOLO, R-CNN, etc.)"""
    # Conceptual overview of object detection architectures
    architectures = {
        'R-CNN': 'Region-based CNN - selective search + CNN',
        'Fast R-CNN': 'Improved R-CNN with ROI pooling',
        'Faster R-CNN': 'Region Proposal Network + Fast R-CNN',
        'YOLO': 'You Only Look Once - single stage detector',
        'SSD': 'Single Shot Detector - multi-scale feature maps',
        'RetinaNet': 'Feature Pyramid Network + Focal Loss'
    }
    
    for name, desc in architectures.items():
        print(f"{name}: {desc}")
    
    return architectures

# ============================================================================
# 18. RECOMMENDER SYSTEMS
# ============================================================================
# WHEN TO USE:
# - Personalization (e-commerce, streaming).
# - Ranking items for users.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Cold start):
# - Use content-based filtering for new items.
# - Use popularity-based recommendations for new users.
# - Hybrid approaches.
# ============================================================================

def collaborative_filtering():
    """Collaborative filtering for recommendations"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # User-item rating matrix (users x items)
    ratings = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    
    # Item-based collaborative filtering
    # Calculate item similarity
    item_similarity = cosine_similarity(ratings.T)
    
    print("Item similarity matrix:")
    print(item_similarity)
    
    # User-based collaborative filtering
    user_similarity = cosine_similarity(ratings)
    
    print("\nUser similarity matrix:")
    print(user_similarity)
    
    # Predict rating for user 0, item 2
    def predict_rating(user_idx, item_idx, ratings, similarity):
        # Weighted average of similar users' ratings
        similar_users = similarity[user_idx]
        ratings_by_similar = ratings[:, item_idx]
        
        numerator = np.dot(similar_users, ratings_by_similar)
        denominator = np.sum(np.abs(similar_users))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    predicted = predict_rating(0, 2, ratings, user_similarity)
    print(f"\nPredicted rating for user 0, item 2: {predicted:.2f}")
    
    return item_similarity, user_similarity

def matrix_factorization():
    """Matrix factorization for recommendations"""
    from sklearn.decomposition import NMF
    
    # User-item rating matrix
    ratings = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    
    # Non-negative Matrix Factorization
    n_features = 2
    model = NMF(n_components=n_features, init='random', random_state=42)
    
    # Factorize: R ≈ W * H
    W = model.fit_transform(ratings)  # User features
    H = model.components_              # Item features
    
    # Reconstruct ratings
    reconstructed = np.dot(W, H)
    
    print(f"User features shape: {W.shape}")
    print(f"Item features shape: {H.shape}")
    print(f"Reconstruction error: {model.reconstruction_err_:.4f}")
    
    return W, H, reconstructed

# ============================================================================
# 19. ANOMALY DETECTION
# ============================================================================
# WHEN TO USE:
# - Fraud detection, intrusion detection, system health monitoring.
# - Unlabeled data with rare events.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., High false positives):
# - Use ensemble of detectors (Isolation Forest + Autoencoder).
# - Tune contamination parameter carefully.
# - Use semi-supervised learning if some labels exist.
# ============================================================================

def isolation_forest_example():
    """Isolation Forest for anomaly detection"""
    from sklearn.ensemble import IsolationForest
    
    # Generate normal data and anomalies
    np.random.seed(42)
    X_normal = np.random.randn(200, 2)
    X_anomaly = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.vstack([X_normal, X_anomaly])
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.1,  # Expected proportion of outliers
        random_state=42
    )
    y_pred = iso_forest.fit_predict(X)
    
    # -1 for anomalies, 1 for normal
    n_anomalies = np.sum(y_pred == -1)
    print(f"Detected {n_anomalies} anomalies out of {len(X)} samples")
    
    return iso_forest

def autoencoder_anomaly_detection():
    """Autoencoder for anomaly detection"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Generate data
        X_normal = np.random.randn(1000, 20)
        
        # Build autoencoder
        input_dim = 20
        encoding_dim = 8
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(encoder_input)
        
        # Decoder
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
        
        # Autoencoder model
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train on normal data
        autoencoder.fit(
            X_normal, X_normal,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Detect anomalies based on reconstruction error
        X_test = np.random.randn(100, 20)
        reconstructed = autoencoder.predict(X_test, verbose=0)
        mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)
        
        # Set threshold (e.g., 95th percentile)
        threshold = np.percentile(mse, 95)
        anomalies = mse > threshold
        
        print(f"Detected {np.sum(anomalies)} anomalies")
        
        return autoencoder
    except ImportError:
        print("TensorFlow not installed")

def one_class_svm():
    """One-Class SVM for anomaly detection"""
    from sklearn.svm import OneClassSVM
    
    # Generate normal data
    X_train = np.random.randn(200, 2)
    
    # Train One-Class SVM
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
    ocsvm.fit(X_train)
    
    # Test data (with some anomalies)
    X_test = np.vstack([
        np.random.randn(50, 2),
        np.random.uniform(low=-4, high=4, size=(10, 2))
    ])
    
    y_pred = ocsvm.predict(X_test)
    n_anomalies = np.sum(y_pred == -1)
    
    print(f"One-Class SVM detected {n_anomalies} anomalies")
    
    return ocsvm

# ============================================================================
# 20. IMBALANCED DATASETS
# ============================================================================
# WHEN TO USE:
# - One class is much rarer than others (e.g., 1% fraud).
# - Standard accuracy is misleading.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Overfitting minority):
# - Use SMOTE combined with Tomek links (clean up boundaries).
# - Use cost-sensitive learning (class weights).
# - Collect more data for minority class if possible.
# ============================================================================

def handle_imbalanced_data():
    """Techniques for handling imbalanced datasets"""
    from sklearn.utils import resample
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, weights=[0.9, 0.1], random_state=42
    )
    
    print(f"Original class distribution: {np.bincount(y)}")
    
    # 1. Random Oversampling
    X_minority = X[y == 1]
    y_minority = y[y == 1]
    
    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority,
        n_samples=len(y[y == 0]),
        random_state=42
    )
    
    X_oversampled = np.vstack([X[y == 0], X_minority_upsampled])
    y_oversampled = np.hstack([y[y == 0], y_minority_upsampled])
    
    print(f"After oversampling: {np.bincount(y_oversampled)}")
    
    # 2. Random Undersampling
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority, y_majority,
        n_samples=len(y[y == 1]),
        random_state=42
    )
    
    X_undersampled = np.vstack([X_majority_downsampled, X[y == 1]])
    y_undersampled = np.hstack([y_majority_downsampled, y[y == 1]])
    
    print(f"After undersampling: {np.bincount(y_undersampled)}")
    
    # 3. SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    print(f"After SMOTE: {np.bincount(y_smote)}")
    
    return X_smote, y_smote

def class_weight_handling():
    """Using class weights for imbalanced data"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.class_weight import compute_class_weight
    
    X, y = make_classification(
        n_samples=1000, n_features=20,
        weights=[0.9, 0.1], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # Train with class weights
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# ============================================================================
# 21. MODEL INTERPRETATION AND EXPLAINABILITY
# ============================================================================
# WHEN TO USE:
# - Regulated industries (finance, healthcare).
# - Debugging model behavior.
# - Building trust with stakeholders.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Black box models):
# - Use SHAP or LIME for local explanations.
# - Use simpler surrogate models to approximate complex ones.
# ============================================================================

def feature_importance_analysis():
    """Analyzing feature importance"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Built-in feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature ranking (built-in):")
    for i in range(5):
        print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    # Permutation importance
    perm_importance = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=42
    )
    
    print("\nPermutation importance (top 5):")
    perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
    for i in range(5):
        print(f"{i+1}. Feature {perm_indices[i]}: "
              f"{perm_importance.importances_mean[perm_indices[i]]:.4f}")
    
    return importances, perm_importance

def shap_values_example():
    """SHAP values for model interpretation"""
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        print(f"SHAP values shape: {np.array(shap_values).shape}")
        
        # Get feature importance from SHAP
        if isinstance(shap_values, list):
            shap_importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        print(f"SHAP importance: {shap_importance}")
        
        return shap_values, explainer
    except ImportError:
        print("SHAP library not installed")

def lime_example():
    """LIME (Local Interpretable Model-agnostic Explanations)"""
    try:
        from lime import lime_tabular
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=[f'feature_{i}' for i in range(X.shape[1])],
            class_names=['class_0', 'class_1'],
            mode='classification'
        )
        
        # Explain a single prediction
        i = 0
        exp = explainer.explain_instance(
            X_test[i], model.predict_proba, num_features=5
        )
        
        print(f"LIME explanation for instance {i}:")
        print(exp.as_list())
        
        return exp
    except ImportError:
        print("LIME library not installed")

# ============================================================================
# 22. MODEL DEPLOYMENT AND MLOPS
# ============================================================================
# WHEN TO USE:
# - Moving models from notebook to production.
# - Scaling to serve many users.
#
# MITIGATION WHEN BEST PRACTICES FAIL (e.g., Model drift):
# - Implement monitoring for data drift and concept drift.
# - Automate retraining pipelines (CI/CD/CT).
# - Use A/B testing for new models.
# ============================================================================

def save_load_sklearn_model():
    """Saving and loading sklearn models"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    # Train model
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, 'random_forest_model.pkl')
    print("Model saved to random_forest_model.pkl")
    
    # Load model
    loaded_model = joblib.load('random_forest_model.pkl')
    
    # Make predictions
    predictions = loaded_model.predict(X[:5])
    print(f"Predictions: {predictions}")
    
    return loaded_model

def create_sklearn_pipeline():
    """Creating a complete sklearn pipeline"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train pipeline
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    print(f"Pipeline accuracy: {accuracy:.4f}")
    
    # Save pipeline
    joblib.dump(pipeline, 'model_pipeline.pkl')
    
    return pipeline

def model_versioning_concept():
    """Concepts for model versioning and tracking"""
    versioning_strategies = {
        'MLflow': 'Track experiments, parameters, and metrics',
        'DVC': 'Data Version Control - Git for data and models',
        'Weights & Biases': 'Experiment tracking and model management',
        'Neptune.ai': 'Metadata store for MLOps',
        'Model Registry': 'Centralized model store with versioning'
    }
    
    for tool, desc in versioning_strategies.items():
        print(f"{tool}: {desc}")
    
    deployment_strategies = {
        'REST API': 'Flask, FastAPI for serving models',
        'Batch Prediction': 'Scheduled predictions on large datasets',
        'Streaming': 'Real-time predictions on streaming data',
        'Edge Deployment': 'Deploy models on edge devices',
        'Model as Service': 'Cloud-based model serving (AWS, GCP, Azure)'
    }
    
    print("\nDeployment Strategies:")
    for strategy, desc in deployment_strategies.items():
        print(f"{strategy}: {desc}")
    
    return versioning_strategies, deployment_strategies

# ============================================================================
# 23. ADVANCED TOPICS
# ============================================================================
# WHEN TO USE:
# - Specialized problems (generative tasks, complex sequences).
# - Research and development.
# ============================================================================

def gan_concept():
    """Generative Adversarial Networks (GAN) - Conceptual"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Generator
        def build_generator(latent_dim=100):
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_dim=latent_dim),
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(784, activation='tanh'),
                layers.Reshape((28, 28, 1))
            ])
            return model
        
        # Discriminator
        def build_discriminator():
            model = keras.Sequential([
                layers.Flatten(input_shape=(28, 28, 1)),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            return model
        
        generator = build_generator()
        discriminator = build_discriminator()
        
        print("GAN architecture created")
        print(f"Generator parameters: {generator.count_params()}")
        print(f"Discriminator parameters: {discriminator.count_params()}")
        
        return generator, discriminator
    except ImportError:
        print("TensorFlow not installed")

def attention_mechanism_concept():
    """Attention mechanism in neural networks"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Simple attention layer
        class AttentionLayer(layers.Layer):
            def __init__(self):
                super(AttentionLayer, self).__init__()
            
            def call(self, inputs):
                # inputs shape: (batch_size, time_steps, features)
                attention_weights = layers.Dense(1, activation='tanh')(inputs)
                attention_weights = layers.Softmax(axis=1)(attention_weights)
                
                # Weighted sum
                context = attention_weights * inputs
                context = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(context)
                
                return context
        
        # Example usage in a model
        input_layer = layers.Input(shape=(10, 64))  # 10 time steps, 64 features
        lstm_out = layers.LSTM(128, return_sequences=True)(input_layer)
        attention_out = AttentionLayer()(lstm_out)
        output = layers.Dense(1, activation='sigmoid')(attention_out)
        
        model = keras.Model(inputs=input_layer, outputs=output)
        
        print("Attention model created")
        print(model.summary())
        
        return model
    except ImportError:
        print("TensorFlow not installed")

def transformer_architecture_concept():
    """Transformer architecture overview"""
    components = {
        'Self-Attention': 'Attention mechanism to relate different positions',
        'Multi-Head Attention': 'Multiple attention mechanisms in parallel',
        'Positional Encoding': 'Add position information to embeddings',
        'Feed-Forward Network': 'Fully connected layers after attention',
        'Layer Normalization': 'Normalize inputs to each layer',
        'Residual Connections': 'Skip connections to help gradient flow'
    }
    
    print("Transformer Components:")
    for component, desc in components.items():
        print(f"{component}: {desc}")
    
    applications = {
        'BERT': 'Bidirectional Encoder Representations from Transformers',
        'GPT': 'Generative Pre-trained Transformer',
        'T5': 'Text-to-Text Transfer Transformer',
        'Vision Transformer': 'Transformers for image classification',
        'DALL-E': 'Text-to-image generation'
    }
    
    print("\nTransformer Applications:")
    for app, desc in applications.items():
        print(f"{app}: {desc}")
    
    return components, applications

def reinforcement_learning_basics():
    """Reinforcement Learning - Q-Learning example"""
    # Simple Q-Learning for grid world
    class QLearning:
        def __init__(self, n_states, n_actions, learning_rate=0.1, 
                     discount_factor=0.95, epsilon=0.1):
            self.q_table = np.zeros((n_states, n_actions))
            self.lr = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon
        
        def choose_action(self, state):
            # Epsilon-greedy policy
            if np.random.random() < self.epsilon:
                return np.random.randint(self.q_table.shape[1])
            else:
                return np.argmax(self.q_table[state])
        
        def update(self, state, action, reward, next_state):
            # Q-learning update rule
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] += self.lr * td_error
    
    # Example usage
    agent = QLearning(n_states=10, n_actions=4)
    
    # Simulate some updates
    for _ in range(100):
        state = np.random.randint(10)
        action = agent.choose_action(state)
        reward = np.random.random()
        next_state = np.random.randint(10)
        agent.update(state, action, reward, next_state)
    
    print("Q-Learning agent trained")
    print(f"Q-table shape: {agent.q_table.shape}")
    
    return agent

# ============================================================================
# 24. COMMON INTERVIEW QUESTIONS - QUICK REFERENCE
# ============================================================================

"""
============================================================================
MACHINE LEARNING INTERVIEW QUESTIONS - SUMMARY
============================================================================

1. BIAS-VARIANCE TRADEOFF
   - Bias: Error from wrong assumptions (underfitting)
   - Variance: Error from sensitivity to training data (overfitting)
   - Goal: Find balance between bias and variance

2. OVERFITTING VS UNDERFITTING
   - Overfitting: Model too complex, memorizes training data
   - Underfitting: Model too simple, can't capture patterns
   - Solutions: Regularization, cross-validation, more data, feature selection

3. REGULARIZATION TECHNIQUES
   - L1 (Lasso): Adds absolute value of coefficients, feature selection
   - L2 (Ridge): Adds squared value of coefficients, shrinks weights
   - Elastic Net: Combination of L1 and L2
   - Dropout: Randomly drop neurons during training (neural networks)

4. GRADIENT DESCENT VARIANTS
   - Batch: Uses all data, slow but stable
   - Stochastic (SGD): Uses one sample, fast but noisy
   - Mini-batch: Uses batch of samples, balanced approach
   - Adam: Adaptive learning rate, combines momentum and RMSprop

5. EVALUATION METRICS
   Classification:
   - Accuracy: (TP + TN) / Total
   - Precision: TP / (TP + FP) - How many predicted positives are correct
   - Recall: TP / (TP + FN) - How many actual positives were found
   - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
   - ROC-AUC: Area under ROC curve
   
   Regression:
   - MSE: Mean Squared Error
   - RMSE: Root Mean Squared Error
   - MAE: Mean Absolute Error
   - R²: Coefficient of determination

6. CROSS-VALIDATION
   - K-Fold: Split data into K folds, train on K-1, test on 1
   - Stratified K-Fold: Preserves class distribution
   - Leave-One-Out: K = n_samples, expensive but unbiased
   - Time Series Split: For temporal data, no shuffling

7. FEATURE ENGINEERING
   - Scaling: StandardScaler, MinMaxScaler, RobustScaler
   - Encoding: Label encoding, One-hot encoding
   - Feature Selection: SelectKBest, RFE, L1-based
   - Feature Creation: Polynomial features, interactions
   - Handling Missing: Mean/median imputation, KNN imputation

8. DIMENSIONALITY REDUCTION
   - PCA: Linear, unsupervised, maximizes variance
   - t-SNE: Non-linear, visualization, preserves local structure
   - LDA: Supervised, maximizes class separation
   - Autoencoders: Neural network-based, non-linear

9. ENSEMBLE METHODS
   - Bagging: Bootstrap + aggregate, reduces variance (Random Forest)
   - Boosting: Sequential, reduces bias (AdaBoost, GradientBoost, XGBoost)
   - Stacking: Train meta-model on predictions of base models
   - Voting: Combine predictions by voting or averaging

10. IMBALANCED DATA
    - Oversampling: SMOTE, random oversampling
    - Undersampling: Random undersampling
    - Class weights: Penalize majority class more
    - Evaluation: Use precision, recall, F1, ROC-AUC (not accuracy)

11. NEURAL NETWORK CONCEPTS
    - Activation Functions: ReLU, Sigmoid, Tanh, Softmax
    - Backpropagation: Chain rule to compute gradients
    - Optimizers: SGD, Adam, RMSprop, Adagrad
    - Batch Normalization: Normalize inputs to each layer
    - Dropout: Prevent overfitting by dropping neurons

12. CNN (Convolutional Neural Networks)
    - Convolution: Apply filters to extract features
    - Pooling: Downsample to reduce dimensions (Max, Average)
    - Fully Connected: Classification at the end
    - Applications: Image classification, object detection

13. RNN (Recurrent Neural Networks)
    - LSTM: Long Short-Term Memory, handles long sequences
    - GRU: Gated Recurrent Unit, simpler than LSTM
    - Bidirectional: Process sequence in both directions
    - Applications: NLP, time series, speech recognition

14. TRANSFORMERS
    - Self-Attention: Relate different positions in sequence
    - Multi-Head Attention: Multiple attention in parallel
    - Positional Encoding: Add position information
    - Applications: BERT, GPT, Vision Transformers

15. TRANSFER LEARNING
    - Use pretrained model on new task
    - Freeze base layers, train only top layers
    - Fine-tuning: Unfreeze some layers and continue training
    - Benefits: Less data needed, faster training

16. HYPERPARAMETER TUNING
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling from parameter space
    - Bayesian Optimization: Use probabilistic model to guide search
    - Halving Grid Search: Successively halve candidates

17. CLUSTERING ALGORITHMS
    - K-Means: Partition-based, requires K
    - Hierarchical: Agglomerative or divisive, dendrograms
    - DBSCAN: Density-based, finds arbitrary shapes, detects outliers
    - Gaussian Mixture: Probabilistic, soft clustering

18. RECOMMENDER SYSTEMS
    - Collaborative Filtering: User-based or item-based similarity
    - Content-Based: Use item features
    - Matrix Factorization: Decompose rating matrix (SVD, NMF)
    - Hybrid: Combine multiple approaches

19. TIME SERIES
    - Components: Trend, Seasonality, Residual
    - ARIMA: AutoRegressive Integrated Moving Average
    - Prophet: Additive model by Facebook
    - LSTM: Deep learning for sequences

20. MODEL DEPLOYMENT
    - Save/Load: joblib, pickle for sklearn; SavedModel for TensorFlow
    - API: Flask, FastAPI for REST endpoints
    - Containerization: Docker for consistent environment
    - Monitoring: Track model performance, data drift
    - A/B Testing: Compare models in production
"""

# ============================================================================
# 25. PRACTICAL CODING PROBLEMS
# ============================================================================

def implement_train_test_split():
    """Implement train-test split from scratch"""
    def custom_train_test_split(X, y, test_size=0.2, random_state=None):
        if random_state:
            np.random.seed(random_state)
        
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        # Random indices
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
        else:
            X_train = X[train_indices]
            X_test = X[test_indices]
        
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
        else:
            y_train = y[train_indices]
            y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    # Test
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def implement_confusion_matrix():
    """Implement confusion matrix from scratch"""
    def custom_confusion_matrix(y_true, y_pred, n_classes=None):
        if n_classes is None:
            n_classes = max(max(y_true), max(y_pred)) + 1
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1
        
        return cm
    
    # Test
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    
    cm = custom_confusion_matrix(y_true, y_pred, n_classes=2)
    print(f"Confusion Matrix:\n{cm}")
    
    # Calculate metrics from confusion matrix
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return cm

def implement_cross_validation():
    """Implement K-Fold cross-validation from scratch"""
    def custom_kfold_cv(X, y, model, n_splits=5):
        n_samples = len(X)
        fold_size = n_samples // n_splits
        scores = []
        
        for i in range(n_splits):
            # Split data
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
            
            test_indices = list(range(test_start, test_end))
            train_indices = list(range(0, test_start)) + list(range(test_end, n_samples))
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            # Train and evaluate
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            scores.append(score)
            
            print(f"Fold {i+1}: {score:.4f}")
        
        print(f"Mean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        return scores
    
    # Test
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    
    scores = custom_kfold_cv(X, y, model, n_splits=5)
    
    return scores

def implement_standardization():
    """Implement StandardScaler from scratch"""
    class CustomStandardScaler:
        def __init__(self):
            self.mean_ = None
            self.std_ = None
        
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            return self
        
        def transform(self, X):
            return (X - self.mean_) / self.std_
        
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
        
        def inverse_transform(self, X):
            return X * self.std_ + self.mean_
    
    # Test
    X = np.random.randn(100, 5) * 10 + 50
    
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Original mean: {X.mean(axis=0)}")
    print(f"Scaled mean: {X_scaled.mean(axis=0)}")
    print(f"Scaled std: {X_scaled.std(axis=0)}")
    
    return scaler

def implement_precision_recall():
    """Calculate precision and recall from scratch"""
    def custom_precision_recall(y_true, y_pred):
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate TP, FP, FN
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        
        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN
        }
    
    # Test
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    
    metrics = custom_precision_recall(y_true, y_pred)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    return metrics

def implement_cosine_similarity():
    """Implement cosine similarity from scratch"""
    def custom_cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        
        return dot_product / (norm_v1 * norm_v2)
    
    # Test
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    similarity = custom_cosine_similarity(v1, v2)
    print(f"Cosine similarity: {similarity:.4f}")
    
    # For matrices
    def cosine_similarity_matrix(X):
        # Normalize rows
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        # Compute dot product
        return np.dot(X_normalized, X_normalized.T)
    
    X = np.random.rand(5, 10)
    sim_matrix = cosine_similarity_matrix(X)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    return similarity, sim_matrix

def implement_euclidean_distance():
    """Implement Euclidean distance from scratch"""
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def euclidean_distance_matrix(X):
        n = X.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                distances[i, j] = euclidean_distance(X[i], X[j])
        
        return distances
    
    # Vectorized version (faster)
    def euclidean_distance_matrix_vectorized(X):
        # X: (n_samples, n_features)
        # Compute ||X||^2
        X_squared = np.sum(X**2, axis=1, keepdims=True)
        # Compute distance matrix using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        distances_squared = X_squared + X_squared.T - 2 * np.dot(X, X.T)
        distances_squared = np.maximum(distances_squared, 0)  # Handle numerical errors
        return np.sqrt(distances_squared)
    
    # Test
    X = np.random.rand(100, 10)
    
    import time
    start = time.time()
    dist_matrix = euclidean_distance_matrix_vectorized(X)
    end = time.time()
    
    print(f"Distance matrix shape: {dist_matrix.shape}")
    print(f"Computation time: {end - start:.4f} seconds")
    
    return dist_matrix

def implement_gradient_descent():
    """Implement gradient descent optimizer from scratch"""
    class GradientDescentOptimizer:
        def __init__(self, learning_rate=0.01):
            self.lr = learning_rate
        
        def update(self, params, grads):
            """Update parameters using gradients"""
            updated_params = {}
            for key in params:
                updated_params[key] = params[key] - self.lr * grads[key]
            return updated_params
    
    class MomentumOptimizer:
        def __init__(self, learning_rate=0.01, momentum=0.9):
            self.lr = learning_rate
            self.momentum = momentum
            self.velocity = {}
        
        def update(self, params, grads):
            if not self.velocity:
                for key in params:
                    self.velocity[key] = np.zeros_like(params[key])
            
            updated_params = {}
            for key in params:
                self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
                updated_params[key] = params[key] + self.velocity[key]
            
            return updated_params
    
    class AdamOptimizer:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.lr = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = {}  # First moment
            self.v = {}  # Second moment
            self.t = 0   # Timestep
        
        def update(self, params, grads):
            if not self.m:
                for key in params:
                    self.m[key] = np.zeros_like(params[key])
                    self.v[key] = np.zeros_like(params[key])
            
            self.t += 1
            updated_params = {}
            
            for key in params:
                # Update biased first moment estimate
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                
                # Update biased second moment estimate
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
                
                # Bias correction
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                updated_params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            return updated_params
    
    # Example usage
    params = {'w': np.array([1.0, 2.0]), 'b': np.array([0.5])}
    grads = {'w': np.array([0.1, 0.2]), 'b': np.array([0.05])}
    
    # SGD
    sgd = GradientDescentOptimizer(learning_rate=0.1)
    updated_params = sgd.update(params, grads)
    print(f"SGD updated params: {updated_params}")
    
    # Adam
    adam = AdamOptimizer(learning_rate=0.01)
    updated_params = adam.update(params, grads)
    print(f"Adam updated params: {updated_params}")
    
    return sgd, adam

# ============================================================================
# 26. REAL-WORLD ML PROBLEMS
# ============================================================================

def customer_churn_prediction_pipeline():
    """Complete pipeline for customer churn prediction"""
    # Simulated customer data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'tenure': np.random.randint(0, 60, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['avg_monthly_charges'] = df['total_charges'] / (df['tenure'] + 1)
    df['charges_to_tenure_ratio'] = df['monthly_charges'] / (df['tenure'] + 1)
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['contract_type', 'internet_service'], drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('churn', axis=1)
    y = df_encoded['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("Churn Prediction Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())
    
    return model, scaler, feature_importance

def fraud_detection_pipeline():
    """Pipeline for fraud detection with imbalanced data"""
    # Generate imbalanced fraud data
    X, y = make_classification(
        n_samples=10000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        weights=[0.98, 0.02],  # 2% fraud
        random_state=42
    )
    
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {np.bincount(y_train_resampled)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train_scaled, y_train_resampled)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nFraud Detection Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return model, scaler

def house_price_prediction_pipeline():
    """Pipeline for house price prediction"""
    # Generate house price data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'sqft': np.random.randint(800, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Target: price (simplified formula)
    df['price'] = (
        df['sqft'] * 200 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 -
        df['age'] * 1000 +
        df['location_score'] * 20000 +
        np.random.normal(0, 20000, n_samples)
    )
    
    # Feature engineering
    df['sqft_per_bedroom'] = df['sqft'] / df['bedrooms']
    df['bathroom_to_bedroom_ratio'] = df['bathrooms'] / df['bedrooms']
    
    # Polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df.drop('price', axis=1))
    
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with regularization
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=10.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("House Price Prediction Results:")
    print(f"MSE: ${mse:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, scaler, poly

# ============================================================================
# END OF COMPREHENSIVE ML INTERVIEW GUIDE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MACHINE LEARNING INTERVIEW PREPARATION GUIDE")
    print("=" * 80)
    print("\nThis guide covers all essential ML concepts for interviews.")
    print("Run individual functions to test specific concepts.")
    print("\nGood luck with your interview!")
    print("=" * 80)
