# src/train.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from preprocess import load_data
from dataset import TweetDataset
from model import get_model, get_tokenizer

# =========================
# 1. Load and reduce data
# =========================
print("Loading data...")

df = load_data("data/twitter.csv")

# 🔥 IMPORTANT: reduce dataset size (avoid freeze)
df = df.sample(5000)

texts = df['text'].tolist()
labels = df['label'].tolist()

print(f"Dataset size: {len(texts)}")

# =========================
# 2. Tokenizer + Model
# =========================
print("Loading tokenizer and model...")

tokenizer = get_tokenizer()
model = get_model()

# =========================
# 3. Dataset + Dataloader
# =========================
print("Creating dataset...")

dataset = TweetDataset(texts, labels, tokenizer)

loader = DataLoader(
    dataset,
    batch_size=8,        # 🔥 smaller batch for CPU
    shuffle=True,
    num_workers=0        # ⚡ faster loading
)

print("Dataloader ready")

# =========================
# 4. Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# =========================
# 5. Training loop
# =========================
for epoch in range(2):
    print(f"\n🚀 Starting Epoch {epoch}")

    model.train()
    total_loss = 0

    for i, batch in enumerate(loader):
        print(f"Batch {i}")  # 👈 helps you see progress

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"✅ Epoch {epoch} Loss: {total_loss}")

# =========================
# 6. Save model
# =========================
torch.save(model.state_dict(), "bert_model.pt")

print("\n🎉 Training Complete! Model saved as bert_model.pt")