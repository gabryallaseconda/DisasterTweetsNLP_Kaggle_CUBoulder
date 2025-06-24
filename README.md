#  Model Analysis: Disaster Tweets NLP (Kaggle)

## Repository Structure & Purpose
- **Goal:** Classify disaster-related tweets (Kaggle competition).
- **Main Notebooks:**
  - `Exploration.ipynb`: Exploratory data analysis (EDA).
  - `Training.ipynb`: Data preprocessing, model training, and inference.
  - `submission.ipynb`: Contains cells from the other two notebooks for submission.
  - `Full_Project_Notebook.ipynb`: Contain the final summary of all the previous.
---

## Model Pipeline: Step-by-Step

### 1. Data Import & EDA
   - Loads training and test data from CSV files.
   - Performs EDA: visualizes distributions, inspects samples, etc.

### 2. Preprocessing
   - **Lowercasing:** Converts all text to lowercase.
   - **Entity, URL, and Punctuation Removal:** Cleans text by removing mentions, hashtags, URLs, and punctuation.
   - **Spell Correction:** Uses SymSpell for correcting spelling errors.
   - **Filling Missing Data:**
     - Extracts keywords using a transformer model (DistilBERT).
     - Fills missing locations using named entity recognition (spaCy).
   - **Lemmatization:** Reduces words to their base form.
   - **Stopword Removal:** Removes common stopwords.

### 3. Feature Engineering
   - Embeds three fields: `keyword`, `location`, and `text` using a pretrained embedding model.

### 4. Model Architecture
   - **Embedding:** Uses a pretrained model from TensorFlow Hub: `https://tfhub.dev/google/nnlm-en-dim50/2` (NNLM English, 50-dimensional).
   - **DistilBERT:** Used for extracting semantic features and keyword extraction (`distilbert-base-nli-mean-tokens` via SentenceTransformers).
   - **Neural Network:** Three parallel dense layers (one for each field), concatenated, followed by more dense layers and dropout, ending with a sigmoid output for binary classification.

### 5. Training
   - Uses SGD optimizer with learning rate decay and early stopping.
   - Trains on the embedded features with binary cross-entropy loss.

### 6. Inference & Submission
   - Embeds test data, predicts probabilities, thresholds at 0.5, and writes results to `submission.csv`.

---

**Summary:**  
The pipeline combines advanced preprocessing (including spell correction and entity handling), semantic feature extraction with DistilBERT, and efficient embedding with a pretrained NNLM model, followed by a custom neural network for classification. DistilBERT was chosen for its balance of speed and performance, and the NNLM model provided robust, ready-to-use embeddings.

