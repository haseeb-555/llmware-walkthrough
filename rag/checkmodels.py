from llmware.models import ModelCatalog

# Load available embedding models
embedding_models = ModelCatalog().list_embedding_models()

# Print them out
for i, model in enumerate(embedding_models):
    print(f"🔢 [{i}] {model}")
