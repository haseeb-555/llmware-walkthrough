import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.setup import Setup
from llmware.status import Status
from llmware.models import ModelCatalog
from llmware.configs import LLMWareConfig, MilvusConfig

from importlib import util

# Check dependencies
if not util.find_spec("torch") or not util.find_spec("transformers"):
    print("\nPlease install transformers and torch to run this example:"
          "\n`pip install torch`"
          "\n`pip install transformers`")

if not (util.find_spec("chromadb") or util.find_spec("pymilvus") or util.find_spec("lancedb") or util.find_spec("faiss")):
    print("\nPlease install a vector DB driver like chromadb, pymilvus, lancedb or faiss.")

def setup_library(library_name):
    print(f"\nCreating library: {library_name}")
    library = Library().create_new_library(library_name)

    embedding_record = library.get_embedding_status()
    print("Embedding record - before embedding:", embedding_record)

    # 🔧 Custom Folder Path for Your PDFs
    input_folder = os.path.join(os.getcwd(), r"C:\Users\Shaheen sultana\OneDrive\Desktop\H36\llmware\llmware-walkthrough\myfolder")
    print(f"\nUsing PDF files from: {input_folder}")

    # Add your own PDF files here instead of sample files
    library.add_files(input_folder_path=input_folder,
                      chunk_size=400, max_chunk_size=600, smart_chunking=1)

    return library


def install_vector_embeddings(library, embedding_model_name):
    library_name = library.library_name
    vector_db = LLMWareConfig().get_vector_db()

    print(f"\nStarting embedding: Library = {library_name}, Vector DB = {vector_db}, Model = {embedding_model_name}")

    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db, batch_size=100)

    update = Status().get_embedding_status(library_name, embedding_model_name)
    print("Embeddings complete - status check:", update)

    # 🔍 Custom Query: "tell about svm"
    sample_query = "tell about svm"
    print(f"\nRunning semantic/vector query: '{sample_query}'")

    query_results = Query(library).semantic_query(sample_query, result_count=20)

    for i, entry in enumerate(query_results):
        text = entry["text"]
        document_source = entry["file_source"]
        page_num = entry["page_num"]
        vector_distance = entry["distance"]

        if len(text) > 125:
            text = text[:125] + " ..."

        print(f"\nResult {i} - Document: {document_source} - Page: {page_num} - Distance: {vector_distance}")
        print("Text sample:", text)

    embedding_record = library.get_embedding_status()
    print("\nEmbedding record - after:", embedding_record)


if __name__ == "__main__":
    LLMWareConfig().set_active_db("sqlite")
    MilvusConfig().set_config("lite", True)
    LLMWareConfig().set_vector_db("chromadb")

    library = setup_library("svm_library")

    embedding_model = "mini-lm-sbert"  # Make sure torch + transformers are installed

    install_vector_embeddings(library, embedding_model)
