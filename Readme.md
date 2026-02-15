# CSL7640 Problem 4: Text Classification (Sports vs. Politics)

**Student Name:** Kalkar Tejas Prasad  
**Roll No:** B23CM1051  
**Course:** Natural Language Understanding  

---

## 1. Project Overview
This project is dedicated to the task of binary text classification. The objective is to design a system. This system reads a text document and classifies it into one of two categories: **Sports** or **Politics**. 

The solution is implemented using Python and Scikit-Learn. I have used **TF-IDF (Term Frequency-Inverse Document Frequency)** for the feature representation. It is better than Bag of Words because it reduces the weight of common words. To ensure a robust analysis is there, I have compared **10 different Machine Learning techniques**.

## 2. Data Collection Strategy
The dataset was not manually downloaded by me. It is fetched programmatically in the code. I used the `requests` library in Python.
- **Source:** BBC News Classification Dataset (Subset).
- **Categories Selected:** Only `sport` and `politics` were filtered.
- **Total Samples:** 928 documents.
- **Class Balance:** 511 Sports docs vs 417 Politics docs.

## 3. Dataset Analysis (EDA)
Before training the models, an extensive analysis was performed. It is to understand the data nature.

### Class Distribution
The dataset is having a nearly balanced distribution. This is good for training.

### Article Lengths
Most articles are having a length between 200 to 500 words. This provides enough tokens for the TF-IDF vectorizer to learn patterns.

### Vocabulary Analysis
I have generated N-grams (Bi-grams) to see the most frequent phrases.
- **Politics:** Words like "Prime Minister", "General Election" are very frequent.
- **Sports:** Words like "World Cup", "Champions League" are dominant.


## 4. Methodology & Techniques
The pipeline implemented is following these steps:
1.  **Preprocessing:** Lowercasing and simple tokenization.
2.  **Feature Extraction:** `TfidfVectorizer` is used with a maximum of 3000 features. Stop words are removed.
3.  **Model Training:** The data is split into 80% Training and 20% Testing sets.

### Models Compared
I have implemented and compared the following algorithms:
1.  **Naive Bayes (Multinomial)** - *Probabilistic*
2.  **Logistic Regression** - *Linear Classifier*
3.  **Random Forest** - *Ensemble*
4.  **Support Vector Machine (Linear SVC)** - *Maximum Margin*
5.  **K-Nearest Neighbors (KNN)** - *Distance based*
6.  **Gradient Boosting** - *Ensemble*
7.  **Decision Tree** - *Tree based*
8.  **Neural Network (MLP)** - *Deep Learning (Basic)*
9.  **Nearest Centroid** - *Simple Baseline*
10. **AdaBoost** - *Boosting*

## 5. Quantitative Results
The results obtained are exceptional. The two categories are highly separable.

| Rank | Algorithm | Accuracy |
| :--- | :--- | :--- |
| **1** | **Naive Bayes** | **100.00%** |
| **1** | **Logistic Regression** | **100.00%** |
| **1** | **SVM (Linear)** | **100.00%** |
| **1** | **Neural Network** | **100.00%** |
| **1** | **Nearest Centroid** | **100.00%** |
| 6 | Random Forest | 99.46% |
| 6 | AdaBoost | 99.46% |
| 9 | Gradient Boosting | 97.31% |
| 10 | Decision Tree | 95.70% |


### Why 100% Accuracy?
To verify if the 100% accuracy is real or a bug, I plotted the **t-SNE** clusters. As seen in the image below, the two classes (Red and Blue) are completely separated in the vector space. There is no overlap. This is why linear models like SVM perform perfectly.


## 6. Limitations of the System
Even though the accuracy is 100%, there are some limitations observed by me:
1.  **Dataset Size:** The dataset is small (only 928 samples). It might not generalize to millions of articles.
2.  **Domain Specificity:** It is trained on BBC news. It might fail on tweets or informal text.
3.  **Static Vocabulary:** The model uses a fixed vocabulary of 3000 words. If new words appear in future news, it will ignore them.

## 7. How to Run the Code
You can run the script using Python 3. It requires the libraries: `scikit-learn`, `pandas`, `seaborn`, `matplotlib`.

```bash
# To run the classifier on a test file
python B23CM1051_prob4.py my_test_file.txt
