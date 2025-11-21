# AIINTERN
A complete AI internship project containing four tasks: data preprocessing, machine learning modeling, natural language processing, and image classification using deep learning. Each task includes fully functional Python scripts, clean pipelines, and documented outputs.
## 📂 Project Structure

── Task1_Data_Preprocessing/
│ └── task1_preprocessing.py
│
├── Task2_ML_Model/
│ └── task2_ml_model.py
│
├── Task3_NLP/
│ ├── task3_nlp_sentiment.py
│ └── sample_output.txt
│
├── Task4_Image_Classification/
│ ├── task4_mnist_cnn.py
│ └── mnist_training_history.png
│
└── README.md


---

# 📌 Task 1 — Data Preprocessing (Pandas + Scikit-Learn)

**Features included:**
- Load & clean dataset  
- Handle missing values  
- Encode categorical features  
- Scale numeric features  
- Feature engineering  
- Export cleaned dataset  

### 🔧 Run:
```bash
python task1_preprocessing.py

📌 Task 2 — Machine Learning Model (Decision Tree & Logistic Regression)

Includes:

Decision Tree with GridSearchCV

Logistic Regression

Cross-validation

Accuracy, classification report, confusion matrix

Auto-save best model

🔧 Run:
python task2_ml_model.py

📌 Task 3 — NLP (Sentiment Analysis)

Includes two powerful NLP methods:

✔️ VADER (rule-based sentiment)
✔️ Supervised ML Model (TF-IDF + LogisticRegression)
✔️ spaCy text cleaning + lemmatization
🔧 Run:
python task3_nlp_sentiment.py

📝 Example Output:
VADER results
Supervised model accuracy: 0.87
spaCy cleaned text: love product exceed expectation

📌 Task 4 — Image Classification (CNN using TensorFlow)

Features:

MNIST dataset loading

CNN with Conv2D, MaxPooling, Dropout

Training curves

Accuracy report

Save model + sample predictions

🔧 Run:
python task4_mnist_cnn.py

🛠 Installation

Install all dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn nltk spacy tensorflow


Download required NLP data:

import nltk
nltk.download('vader_lexicon')
nltk.download('movie_reviews')
nltk.download('punkt')


Download spaCy English model:

python -m spacy download en_core_web_sm

💡 About

This repository showcases fundamental AI skills:

✔ Data preprocessing
✔ Machine learning
✔ Natural language processing
✔ Deep learning
