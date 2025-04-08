from llmware.agents import LLMfx
from llmware.library import Library
from llmware.retrieval import Query
from llmware.configs import LLMWareConfig


def multistep_analysis():

    # Step 1: Load the previously created library
    LLMWareConfig().set_active_db("sqlite")
    my_lib = Library().load_library("microsoft_history_0823_1")

    # Step 2: Search for references to "ibm"
    query = "ibm"
    search_results = Query(my_lib).text_query(query)

    # Step 3: Create agent and load analysis tools
    agent = LLMfx()
    agent.load_tool_list(["sentiment", "emotions", "topic", "tags", "ner", "answer"])
    agent.load_work(search_results)

    # Step 4: Run sentiment analysis
    while True:
        agent.sentiment()
        if not agent.increment_work_iteration():
            break

    # Step 5: Follow-up on negative sentiment
    follow_up_list = agent.follow_up_list(key="sentiment", value="negative")

    for job_index in follow_up_list:
        agent.set_work_iteration(job_index)
        agent.exec_multitool_function_call(["tags", "emotions", "topics", "ner"])
        agent.answer("What is a brief summary?", key="summary")

    # Step 6: Print final report
    my_report = agent.show_report(follow_up_list)
    activity_summary = agent.activity_summary()

    print("\n📘 Final Analysis Report:\n")
    for entry in my_report:
        print(entry)

    print("\n📊 Activity Summary:\n")
    print(activity_summary)

    return my_report


if __name__ == "__main__":
    multistep_analysis()
