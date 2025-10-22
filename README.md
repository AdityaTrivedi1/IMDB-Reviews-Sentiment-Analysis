# 📘 IMDB Reviews Sentiment Analysis  

## 📄 Overview  
This project performs **sentiment analysis** on IMDB movie reviews to classify text as **positive** or **negative**. It leverages **Word2Vec embeddings** and a **Bidirectional LSTM network** to capture contextual relationships within text, achieving strong performance on natural language data.  

The model is built using **TensorFlow/Keras**, trained on processed IMDB data, and evaluated through standard classification metrics such as **accuracy, F1-score, precision, recall**, and **ROC-AUC**.  

---

## 🧠 Methodology  

### 1. Data Loading  
The dataset used is the **IMDB Movie Reviews Dataset**, containing labeled text reviews (positive/negative).  

### 2. Data Preprocessing  
Key preprocessing steps include:  
- Text normalization (lowercasing, punctuation removal)  
- Tokenization and sequencing  
- Padding to equal sequence length  
- Train-test split using `train_test_split()`  

### 3. Word Embedding  
A **pretrained Word2Vec** model (via Gensim) is loaded to map words to dense vector representations.  
An embedding matrix is built from these vectors to initialize the Keras Embedding layer.  

### 4. Model Architecture  
A **Bidirectional LSTM** model captures contextual dependencies from both directions in the text.  

Model structure:  
```python
Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)
Dropout(0.5)
Bidirectional(LSTM(48))
Dropout(0.6)
Dense(1, activation="sigmoid")
```

### 5. Training  
The model is compiled using:
- **Loss**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  
- **Callback**: EarlyStopping for regularization  

### 6. Evaluation  
Performance metrics include:
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC-AUC  

Visualizations are generated using **Plotly** to interpret model results and performance trends.  

---

## 📊 Results  
The LSTM model demonstrates high sentiment classification performance, showing strong generalization on unseen test data (exact numbers depend on training outcomes).  
Metrics such as **accuracy >85%** and **ROC-AUC ≈0.9** are typical for this setup.  

---

## 🧩 Dependencies  

Ensure you have the following libraries installed:  

```bash
pip install numpy pandas scikit-learn tensorflow gensim plotly joblib
```

Python version: **3.8+**
---

## ⚙️ Setup & Usage  

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/IMDB_Reviews_Sentiment_Analysis.git
   cd IMDB_Reviews_Sentiment_Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Open the notebook**
    ```bash
    jupyter notebook IMDB_Reviews_Sentiment_Analysis.ipynb
    ```

## 🧾 References  

- IMDB Dataset: [Kaggle IMDB Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
