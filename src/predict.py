import torch
from src.model import get_model, get_tokenizer
from src.preprocess import clean_text
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer()
model = get_model()

# 🔥 Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="swayamkr03/bert-sentiment",  # 👈 your repo
    filename="bert_model.pt"
)

model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()

label_map = {
    0: "Negative",
    1: "Positive"
}

def predict(text):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return label_map[pred], confidence, probs[0].tolist()