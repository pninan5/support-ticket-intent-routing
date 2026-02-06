import re
import joblib
import numpy as np

MODEL_PATH = "C:/Users/patri/OneDrive/Desktop/JobHunt2025/support_ticket_nlp/models/intent_router.joblib"

def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"http\S+|www\.\S+", " __url__ ", s)
    s = re.sub(r"\S+@\S+", " __email__ ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def predict_topk(text: str, top_k: int = 3):
    model = joblib.load(MODEL_PATH)
    text_c = clean_text(text)
    probs = model.predict_proba([text_c])[0]
    classes = model.classes_
    top = np.argsort(probs)[-top_k:][::-1]
    return [(classes[i], float(probs[i])) for i in top]

if __name__ == "__main__":
    examples = [
        "I want to cancel my order, can you help?",
        "I forgot my password and cannot log in",
        "How long does delivery take?",
        "Please change the shipping address for my order",
    ]

    for ex in examples:
        print("\nText:", ex)
        print("Top intents:", predict_topk(ex, top_k=3))
