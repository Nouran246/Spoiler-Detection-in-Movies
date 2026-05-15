from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np

from transformers import (
    AutoModel,
    DistilBertTokenizerFast
)

from transformers.modeling_outputs import SequenceClassifierOutput

# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)

# =========================================================
# Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Tokenizer
# =========================================================

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-cased"
)

# =========================================================
# Base DistilBERT
# =========================================================

distilBERT = AutoModel.from_pretrained(
    "distilbert-base-cased"
)

# =========================================================
# Custom Model
# =========================================================

class DistilBERTModified(nn.Module):

    def __init__(self):

        super().__init__()

        self.distilbert = distilBERT

        self.dropout1 = nn.Dropout(p=0.1)
        self.dense1 = nn.Linear(770, 770)

        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.1)
        self.dense2 = nn.Linear(770, 256)

        self.relu2 = nn.ReLU()

        self.dropout3 = nn.Dropout(p=0.1)
        self.dense3 = nn.Linear(256, 32)

        self.relu3 = nn.ReLU()

        self.dropout4 = nn.Dropout(p=0.1)
        self.dense4 = nn.Linear(32, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        _rel_pos_sentences,
        _len_sentence_logs,
        labels=None
    ):

        output = self.distilbert(
            input_ids,
            attention_mask=attention_mask
        )

        output = output[0]
        output = output[:, 0]

        output = torch.hstack((
            output,
            _rel_pos_sentences.reshape(-1, 1),
            _len_sentence_logs.reshape(-1, 1)
        ))

        output = self.relu1(
            self.dense1(
                self.dropout1(output)
            )
        )

        output = self.relu2(
            self.dense2(
                self.dropout2(output)
            )
        )

        output = self.relu3(
            self.dense3(
                self.dropout3(output)
            )
        )

        logits = self.dense4(
            self.dropout4(output)
        )

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

# =========================================================
# Load Model
# =========================================================

model = DistilBERTModified()

model.load_state_dict(
    torch.load(
        "checkpoint.pt",
        map_location=device
    )
)

model.to(device)
model.eval()

# =========================================================
# Prediction Function
# =========================================================

def predict_spoiler(review_text):

    encoded = tokenizer(
        review_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Engineered features
    rel_pos = torch.tensor([0.5]).to(device)

    len_log = torch.tensor([
        np.log(len(review_text) + 1)
    ]).to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            _rel_pos_sentences=rel_pos,
            _len_sentence_logs=len_log
        )

        logits = outputs.logits

        probability = torch.sigmoid(logits).item()

    prediction = 1 if probability > 0.5 else 0

    return prediction, probability

# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        review = request.form["review"]

        pred, prob = predict_spoiler(review)

        confidence = round(prob * 100, 2)

        if pred == 1:
            prediction = "Spoiler Review"
        else:
            prediction = "Non-Spoiler Review"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)