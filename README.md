# 📚 Spoiler Detection in Movies (NLP Web App)

A deep learning–powered web application that detects whether a movie or book review contains **spoilers or not** using a fine-tuned **DistilBERT-based neural network** with additional engineered features.

---

## 🚀 Project Overview

This project classifies text reviews into:

* ✅ Spoiler Review
* ❌ Non-Spoiler Review

It uses a hybrid NLP model:

* Transformer embeddings (DistilBERT)
* Engineered features (text length, relative position)
* Fully connected neural network classifier
* Flask-based web interface

---

## 🧠 Model Architecture

The model is built on top of:

* `distilbert-base-cased` (Transformer backbone)
* Custom classification head:

```
768 (CLS embedding)
+ 2 engineered features
= 770 input features

→ Linear(770 → 770)
→ Linear(770 → 256)
→ Linear(256 → 32)
→ Linear(32 → 1)
→ Sigmoid output
```

---

## 📊 Features Used

* BERT `[CLS]` token embedding
* Relative position feature (`rel_pos`)
* Log of sentence length (`len_log`)

---

## ⚙️ Tech Stack

* Python 🐍
* PyTorch 🔥
* Hugging Face Transformers 🤗
* Flask 🌐
* NumPy

---

## 📁 Project Structure

```
Spoiler-Detection-in-Movies/
│
├── web application/
│   ├── app.py              # Flask backend
│   ├── templates/
│   │   └── index.html     # Frontend UI
│
├── checkpoint.pt          # Trained model weights
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Spoiler-Detection-in-Movies.git
cd Spoiler-Detection-in-Movies
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Flask app

```bash
python app.py
```

---

### 4. Open in browser

```
http://127.0.0.1:5000
```

---

## 🧪 How It Works

1. User enters a review
2. Text is tokenized using DistilBERT tokenizer
3. Model extracts contextual embeddings
4. Engineered features are appended
5. Classifier predicts probability of spoiler
6. Final output is:

   * Spoiler Review (prob ≥ 0.40)
   * Non-Spoiler Review (prob < 0.40)

---

## 📌 Example

### Input:

> “At the end of the movie, he sacrifices himself to save everyone.”

### Output:

```
Spoiler Review
Confidence: 78%
```

---

## ⚠️ Important Notes

* The model is **not retrained at runtime**
* It uses a pre-trained checkpoint (`checkpoint.pt`)
* Performance depends heavily on training dataset quality
* Threshold tuned to **0.40 for best real-world results**

---

## 🧠 Limitations

* May misclassify ambiguous summaries
* Sensitive to short inputs
* Not fully calibrated probability output
* Dataset-dependent behavior

---

## 🔧 Possible Improvements

* Probability calibration (Platt scaling / temperature scaling)
* Better dataset balancing
* Training with full review context
* Explainability (highlight spoiler words)
* Transformer-only head (remove engineered features)

---

## 👩‍💻 Author

Built as part of an AI/NLP learning project focusing on:

* Transformer models
* Text classification
* Real-world deployment using Flask

---

## 📄 License

This project is for educational purposes.

