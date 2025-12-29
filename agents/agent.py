from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from utils.config import config
import tools.retriever_tool as retriever_tool
import json

CONFIDENCE_THRESHOLD = 50

def retrieve_agent(user_query: str):
    model = ChatOpenAI(model=config.chat_model, temperature=0)

    system_prompt = """
     You are a RAG agent. Retrieve context first then answer based only on context.
    If info is not present, say "Not found in documents".
    Never hallucinate.
    """

    agent = create_agent(
        model=model,
        tools=[retriever_tool.retrieve_ans],
        system_prompt=system_prompt
    )

    try:
        result = agent.invoke({"messages":[{"role":"user","content":user_query}]})
        print("\n🤖 -=====++>> Agent Result:", result, "\n")
        tool_output = result["messages"][2].content
        tool_data = json.loads(tool_output)

        contexts = tool_data.get("contexts", [])
        scores   = tool_data.get("scores", [])

        if not contexts:
            return {"messages":[{"role":"assistant","content":"No relevant document found."}]}

        # convert list of scores → percentage
        best_score = max(scores)
        confidence = round(((best_score+1)/2)*100,2)

        if confidence < CONFIDENCE_THRESHOLD:
            return {"messages":[{"role":"assistant",
                "content": f"⚠ Low confidence ({confidence}%). Try rephrasing."}]}

        # -------- join top K docs for summarization ----------
        merged_context = "\n\n".join(contexts[:3])

        final_prompt = f"""
        Answer using ONLY the following context.
        Summarize in your own words. If not found, say "Not present in docs".

        Question: {user_query}

        Context:
        {merged_context}
        """

        answer = model.invoke(final_prompt).content.strip()

        response = f"""
                    **Answer based on documents:**
                    {answer}
                    \n**Confidence:** {confidence}% """

        return {"messages":[{"role":"assistant", "content":response}]}

    except Exception as e:
        return {"messages":[{"role":"assistant","content":f"Error: {str(e)}"}]}
