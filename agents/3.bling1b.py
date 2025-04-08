

"""                             WELCOME TO LLMWARE - BLING & DRAGON Model Zoo Sampler

    Get started with local inferences in minutes ...

    This example is a "hello world" model zoo sampler for LLMWare BLING and DRAGON models, and shows a simple
    repeatable recipe for prompting models locally using a provided set of sample context passages and questions.

    It is designed to be easy to select among different LLMWARE models in the BLING (Best Little Instruct No-GPU)
    and DRAGON (Delivering RAG On ...) model families.

    Also, please feel free to swap out the test passages and questions with your own test set.

                                        BACKGROUND ON BLING + DRAGON

    All of these models have been fine-tuned on complex business, financial and legal materials, with specific
    focus on accurate fact-based question-answering for a context passage.  Key training objectives:

    -- Fact-based - rely upon the grounded truth from the context passage, rather than any 'background' implicit
    knowledge on the subject.  As a result, if no context passage is provided in the prompt, the model will often
    respond with "Not Found" or potentially no response.  This is less fun for chat applications, but extremely
    useful behavior for RAG, Agent and workflow based processes.   Hallucinations are rather low as a result,
    especially when using `temperature=0.0` and `sample=False` options.

    -- Short Answers - the responses will not be expansive in a pleasing dialog/chat oriented way, but rather
    stick to the facts and answer the target question, often times with only a few tokens.   This helps with both
    programmatic workflows to categorize/sort responses, integrate answers into reports and other automated
    aggregations, as well as improving the speed of generation for local inference.

    -- Negative sampling with "Not Found" - the training set includes a well-curated set of negative samples
    in which the question can not be answered from the context passage, and rather than "over-interpret", "fill
    in the gaps" or provide a lengthy apology message, the model will generally respond with the consistent
    "Not Found" message which can be useful in RAG contexts in looking to extract values to identify among
    multiple passages which, if any, can answer the target question.

    -- No prompt instructions expected - the model has been fine-tuned to expect a context passage to read, and then
    answer a question based on it in a fact-based, grounded way.   No need for "You are a helpful assistant" and other
    verbiage.

    BLING models vary between 0.5B - 3.8B parameters and have been specifically designed to run on a CPU, including
    on local laptops.

    DRAGON models vary between 6-9B parameters, and when quantized, can generally run OK on most CPU-based laptops,
    but are designed for optimal use on a GPU or inference server.   These models operate on the same principles
    as the BLING models, making it easy to 'test' with BLING, and then 'upgrade' to DRAGON in shifting into
    production environment for greater accuracy.

    We are always updating the BLING and DRAGON model collection with new models, including improvements in the training
    techniques and experimenting with new base models, and try to keep this script updated as a good
    'hello world' sandbox entry point for testing and evaluation.

    We have trained these models on a wide range of base foundation models to also support preferences among specific
    users and clients for a particular base model.

    We score each of these models on a RAG benchmark test for accuracy and a number of specialized metrics.  Please
    see the example "get_model_benchmarks" for a view of this.

"""


import time
from llmware.prompts import Prompt


def hello_world_questions():

    """ Representative test set - we would recommend running this script as a 'hello world' test on the first
    try that you use a model, and then adapt the content to your own set of context and questions.

    There is nothing special about these questions, and in fact, you will note that many of the models will get
    a couple of answers wrong (especially the ~1B parameter models).   The errors are important insights
    as you evaluate which models to consider for your use case.

    To adapt this test set, just create your own list with dictionary entries and keys 'query', 'answer' and 'context'.

    """


    test_list = [

        {"query": "What is the total amount of the invoice?",
        "answer": "$22,500.00",
        "context": "Services Vendor Inc. \n100 Elm Street Pleasantville, NY \nTO Alpha Inc. 5900 1st Street "
                    "Los Angeles, CA \nDescription Front End Engineering Service $5000.00 \n Back End Engineering"
                    " Service $7500.00 \n Quality Assurance Manager $10,000.00 \n Total Amount $22,500.00 \n"
                    "Make all checks payable to Services Vendor Inc. Payment is due within 30 days."
                    "If you have any questions concerning this invoice, contact Bia Hermes. "
                    "THANK YOU FOR YOUR BUSINESS!  INVOICE INVOICE # 0001 DATE 01/01/2022 FOR Alpha Project P.O. # 1000"},

        {"query": "What was the amount of the trade surplus?",
        "answer": "62.4 billion yen ($416.6 million)",
        "context": "Japan’s September trade balance swings into surplus, surprising expectations"
                    "Japan recorded a trade surplus of 62.4 billion yen ($416.6 million) for September, "
                    "beating expectations from economists polled by Reuters for a trade deficit of 42.5 "
                    "billion yen. Data from Japan’s customs agency revealed that exports in September "
                    "increased 4.3% year on year, while imports slid 16.3% compared to the same period "
                    "last year. According to FactSet, exports to Asia fell for the ninth straight month, "
                    "which reflected ongoing China weakness. Exports were supported by shipments to "
                    "Western markets, FactSet added. — Lim Hui Jie"},

        {"query": "What was Microsoft's revenue in the 3rd quarter?",
        "answer": "$52.9 billion",
        "context": "Microsoft Cloud Strength Drives Third Quarter Results \nREDMOND, Wash. — April 25, 2023 — "
                    "Microsoft Corp. today announced the following results for the quarter ended March 31, 2023,"
                    " as compared to the corresponding period of last fiscal year:\n· Revenue was $52.9 billion"
                    " and increased 7% (up 10% in constant currency)\n· Operating income was $22.4 billion "
                    "and increased 10% (up 15% in constant currency)\n· Net income was $18.3 billion and "
                    "increased 9% (up 14% in constant currency)\n· Diluted earnings per share was $2.45 "
                    "and increased 10% (up 14% in constant currency).\n"},

        {"query": "When did the LISP machine market collapse?",
        "answer": "1987.",
        "context": "The attendees became the leaders of AI research in the 1960s."
                    "  They and their students produced programs that the press described as 'astonishing': "
                    "computers were learning checkers strategies, solving word problems in algebra, "
                    "proving logical theorems and speaking English.  By the middle of the 1960s, research in "
                    "the U.S. was heavily funded by the Department of Defense and laboratories had been "
                    "established around the world. Herbert Simon predicted, 'machines will be capable, "
                    "within twenty years, of doing any work a man can do'.  Marvin Minsky agreed, writing, "
                    "'within a generation ... the problem of creating 'artificial intelligence' will "
                    "substantially be solved'. They had, however, underestimated the difficulty of the problem.  "
                    "Both the U.S. and British governments cut off exploratory research in response "
                    "to the criticism of Sir James Lighthill and ongoing pressure from the US Congress "
                    "to fund more productive projects. Minsky's and Papert's book Perceptrons was understood "
                    "as proving that artificial neural networks approach would never be useful for solving "
                    "real-world tasks, thus discrediting the approach altogether.  The 'AI winter', a period "
                    "when obtaining funding for AI projects was difficult, followed.  In the early 1980s, "
                    "AI research was revived by the commercial success of expert systems, a form of AI "
                    "program that simulated the knowledge and analytical skills of human experts. By 1985, "
                    "the market for AI had reached over a billion dollars. At the same time, Japan's fifth "
                    "generation computer project inspired the U.S. and British governments to restore funding "
                    "for academic research. However, beginning with the collapse of the Lisp Machine market "
                    "in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began."}
    ]
    
    
    return test_list
def llmware_bling_dragon_hello_world (model_name):

    """ Simple inference loop that loads a model and runs through a series of test questions. """

    t0 = time.time()
    test_list = hello_world_questions()

    print(f"\n > Loading Model: {model_name}...")

    #   please note that by default, we recommend setting temperature=0.0 and sample=False for fact-based RAG
    prompter = Prompt().load_model(model_name, temperature=0.0, sample=False)

    t1 = time.time()
    print(f"\n > Model {model_name} load time: {t1-t0} seconds")
 
    for i, entries in enumerate(test_list):
        print(f"\n{i+1}. Query: {entries['query']}")
     
        # run the prompt
        output = prompter.prompt_main(entries["query"],context=entries["context"], prompt_name="default_with_context")

        llm_response = output["llm_response"].strip("\n")
        print(f"LLM Response: {llm_response}")
        print(f"Gold Answer: {entries['answer']}")
        print(f"LLM Usage: {output['usage']}")

    t2 = time.time()
    print(f"\nTotal processing time: {t2-t1} seconds")

    return 0


if __name__ == "__main__":

    bling_pytorch = [

        #   pytorch models - will run fast on GPU, and smaller ones good for CPU only
        #   note: you will need to install pytorch and transformers to pull and access these models

        "llmware/bling-1b-0.1",
        "llmware/bling-tiny-llama-v0",
        "llmware/bling-1.4b-0.1",
        "llmware/bling-falcon-1b-0.1",
        "llmware/bling-cerebras-1.3b-0.1",
        "llmware/bling-sheared-llama-1.3b-0.1",
        "llmware/bling-sheared-llama-2.7b-0.1",
        "llmware/bling-red-pajamas-3b-0.1",
        "llmware/bling-stable-lm-3b-4e1t-v0",
        "llmware/bling-phi-3",
        "llmware/bling-phi-3.5"
    ]

    dragon_pytorch = [

        #   pytorch models - intended for GPU server use - will require pytorch, transformers, and in some cases,
        #   other dependencies (einops, flash_attn).

        "llmware/dragon-mistral-7b-v0",
        "llmware/dragon-yi-6b-v0",
        "llmware/dragon-qwen-7b",
        "llmware/dragon-llama-7b-v0",
        "llmware/dragon-mistral-0.3",
        "llmware/dragon-llama-3.1",
        "llmware/dragon-deci-7b-v0"
    ]

    bling_gguf = [

        #   smaller cpu-oriented models - optimal for running on a CPU

        "bling-phi-3.5-gguf",       # **NEW** - phi-3.5 (3.8b)
        "bling-answer-tool",        # this is quantized bling-tiny-llama (1.1b)
        "bling-qwen-0.5b-gguf",     # **NEW** - qwen2 (0.5b)
        "bling-qwen-1.5b-gguf",     # **NEW** - qwen2 (1.5b)
        "bling-stablelm-3b-tool",   # quantized bling-stablelm-3b (2.7b)
        "bling-phi-3-gguf",         # quantized phi-3 (3.8b)
        "bling-phi-2-gguf",         # quantized phi-2 (2.7b)
        ]

    dragon_gguf = [

        #   larger models - 6b - 9b

        "dragon-yi-answer-tool",     # quantized yi-6b (v1) (6b)
        "dragon-llama-answer-tool",
        "dragon-mistral-answer-tool",
        "dragon-qwen-7b-gguf",          # **NEW** qwen2-7b (7b)
        "dragon-yi-9b-gguf",            # **NEW** yi-9b (8.8b)
        "dragon-llama-3.1-gguf",
        "dragon-mistral-0.3-gguf"

    ]

    #   for most use cases, we would recommend using the GGUF for faster inference
    #   NEW - if you are running on a Windows machine, then try substituting for one of the following:
    #    -- "bling-tiny-llama-ov" -> uses OpenVino model version - requires `pip install openvino` and `pip install openvino_genai`  
    #    -- "bling-tiny-llama-onnx" -> uses ONNX model version - requires `pip install onnxruntime_genai`  
    
    my_model = bling_gguf[1]

    llmware_bling_dragon_hello_world(my_model)