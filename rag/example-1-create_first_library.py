import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.configs import LLMWareConfig

def parsing_documents_into_library(library_name, custom_folder_path):

    print(f"\n📚 Parsing Files into Library: {library_name}")

    # Step 1: Create a new library
    library = Library().create_new_library(library_name)
    print(f"✅ Created library: {library_name}")

    # Step 2: Add your custom files
    print(f"📂 Parsing files from: {custom_folder_path}")
    parsing_output = library.add_files(custom_folder_path)
    print(f"✅ Parsing complete: {parsing_output}")

    # Step 3: Show library summary
    card = library.get_library_card()
    print(f"📊 Library Summary → Documents: {card['documents']}, Blocks: {card['blocks']}")

    # Step 4: Run a sample query
    query = Query(library)
    search_text = "explain about support vector machines"  # 🔁 Change this to your actual search intent
    print(f"\n🔍 Query: {search_text}")
    results = query.text_query(search_text, result_count=5)

    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"📄 File: {res['file_source']}")
        print(f"📍 Page: {res['page_num']}")
        print(f"🧠 Text:\n{res['text']}\n")

    return parsing_output


if __name__ == "__main__":
    # Optional: Switch to sqlite if you're not using MongoDB
    LLMWareConfig().set_active_db("sqlite")
    LLMWareConfig().set_config("debug_mode", 2)
    my_custom_path = r"C:\Users\Shaheen sultana\OneDrive\Desktop\H36\llmware\llmware-walkthrough\myfolder"

    # Path to your custom folder with 3 PDFs
    my_custom_path = os.path.join(os.getcwd(), my_custom_path)  # or use full path like "C:/Users/YourName/Desktop/myfolder"

    library_name = "my_custom_library"

    parsing_documents_into_library(library_name, my_custom_path)
