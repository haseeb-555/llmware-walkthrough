from llmware.library import Library
from llmware.retrieval import Query

# Step 1: Create the object
library = Library()

# Step 2: Load the existing library
library.load_library("my_custom_library")

# Step 3: Run query
query_runner = Query(library)

# Your questions
queries = [
    "naive bayes vs svm",
    "decision trees advantages",
    "k-nearest neighbors explanation"
]

# Execute
for q in queries:
    print(f"\n🔍 Query: {q}")
    results = query_runner.query(q, result_count=3)
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"📄 File: {res['file_source']} (Page {res['page_num']})")
        print(f"🧠 Text: {res['text']}\n")
