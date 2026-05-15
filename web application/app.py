from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, DistilBertTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput

# =========================
# Flask App
# =========================
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Tokenizer
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

# =========================
# Model Definition (MUST match training)
# =========================
class DistilBERTModified(nn.Module):
    def __init__(self):
        super().__init__()

        self.distilbert = AutoModel.from_pretrained("distilbert-base-cased")

        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(770, 770)

        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(770, 256)

        self.relu2 = nn.ReLU()

        self.dropout3 = nn.Dropout(0.1)
        self.dense3 = nn.Linear(256, 32)

        self.relu3 = nn.ReLU()

        self.dropout4 = nn.Dropout(0.1)
        self.dense4 = nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask, rel_pos, len_log):

        output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        # FORCE correct dtype + shape
        rel_pos = rel_pos.view(-1, 1).float()
        len_log = len_log.view(-1, 1).float()

        output = torch.cat([output, rel_pos, len_log], dim=1)

        x = self.relu1(self.dense1(self.dropout1(output)))
        x = self.relu2(self.dense2(self.dropout2(x)))
        x = self.relu3(self.dense3(self.dropout3(x)))

        logits = self.dense4(self.dropout4(x))

        return SequenceClassifierOutput(logits=logits)


# =========================
# Load Model
# =========================
model = DistilBERTModified()

model.load_state_dict(
    torch.load("checkpoint.pt", map_location=device)
)

model.to(device)
model.eval()

# =========================
# Spoiler keyword boost
# =========================
SPOILER_KEYWORDS = [
    "dies", "killed", "murder", "death",
    "ending", "final", "reveals", "betrayal",
    "kill", "suicide", "end"
]

# =========================
# Prediction Function
# =========================
def predict_spoiler(text):

    # Normalize input (VERY IMPORTANT)
    text = "Review: " + text.strip()

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    rel_pos = torch.tensor([0.5], dtype=torch.float32, device=device)
    len_log = torch.tensor(
        [np.log(len(text) + 1)],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, rel_pos, len_log)

    logits = outputs.logits

    prob = torch.sigmoid(logits).item()

    # =========================
    # Keyword boost (fix weak predictions)
    # =========================
    lower_text = text.lower()

    if any(word in lower_text for word in SPOILER_KEYWORDS):
        prob += 0.15  # boost spoiler sensitivity

    prob = min(prob, 1.0)

    # =========================
    # Better threshold (important!)
    # =========================
    threshold = 0.40

    pred = 1 if prob > threshold else 0

    return pred, prob


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        review = request.form["review"]

        pred, prob = predict_spoiler(review)

        if prob < 0.40:
            prediction = " Non-Spoiler✅"
        else:
            prediction = "Spoiler⚠"


    return render_template(
        "index.html",
        prediction=prediction,
    )


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)