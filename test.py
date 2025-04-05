from llmware.models import ModelCatalog

model = ModelCatalog().load_model("phi-3-onnx")

response = model.inference("What is the capital of France?")
print(response)
