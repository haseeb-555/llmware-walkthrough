import openvino_genai as ov_genai
import os

# 👉 Set model path (change if different)
model_path = "./TinyLlama-1.1B-Chat-v1.0-openvino-int4"

# 👉 Force CPU backend
os.environ["OPENVINO_DEFAULT_DEVICE"] = "CPU"

# 🔁 Stream output like a chatbot
pipe = ov_genai.LLMPipeline(model_path, "CPU")

# 💬 Infinite chat loop
print("\n🤖 Chatbot Ready! Type 'exit' to quit.")
while True:
    prompt = input("\nYou: ")
    if prompt.strip().lower() == "exit":
        break
    print("TinyLlama:", end=" ")
    pipe.generate(prompt, streamer=lambda x: print(x, end='', flush=True), max_new_tokens=100)
    print()
