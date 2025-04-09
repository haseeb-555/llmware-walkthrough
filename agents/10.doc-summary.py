import os
from llmware.prompts import Prompt

def summarize_custom_document():

    # Your custom PDF path
    fp = r"C:\Users\Shaheen sultana\OneDrive\Desktop\H36\llmware\llmware-walkthrough\myfolder"
    fn = "Lecture-1-Introduction-to-Data-Mining.pdf"

    # Optional: define topic or query to focus the summary
    topic = "data mining introduction"
    query = None

    # Generate summary using the slim-summary tool
    kp = Prompt().summarize_document_fc(fp, fn, topic=topic, query=query, text_only=True, max_batch_cap=100)

    print(f"\nDocument summary completed - {len(kp)} Points\n")
    for i, point in enumerate(kp):
        print(f"{i+1}. {point}")

    return 0

if __name__ == "__main__":
    print("\nSummarizing your custom document...\n")
    summarize_custom_document()
