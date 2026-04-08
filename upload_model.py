from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="bert_model.pt",
    path_in_repo="bert_model.pt",
    repo_id="swayamkr03/bert-sentiment",
    repo_type="model"
)

print("✅ Model uploaded successfully!")