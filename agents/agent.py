from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from utils.config import config
import tools.retriever_tool as retriever_tool
import json

CONFIDENCE_THRESHOLD = 50   # raw cosine score threshold

def retrieve_agent(user_query: str):
    model = ChatOpenAI(model=config.chat_model, temperature=0)
    expanded_query = (
        f"{user_query}. Define, explain, introduction, overview, meaning, basics, concept"
    )
    
    system_prompt = """
    You are a RAG agent. Retrieve context first then answer based only on context.
    If info is not present, say "Not found in documents".
    Never hallucinate.
    """

    agent = create_agent(
        model=model,
        tools=[retriever_tool.retrieve_ans],
        system_prompt=system_prompt,
        context_schema="Retrieve document, then generate answer."
    )

    try:
        # step 1: call agent + retrieve tool result
        result = agent.invoke({"messages":[{"role":"user","content":expanded_query}]})
        tool_output = result["messages"][2].content  # raw tool response

        try: tool_data = json.loads(tool_output)
        except: return {"messages":[{"role":"assistant","content":"⚠ Failed: Tool returned invalid output"}]}

        context = tool_data.get("context", "")
        score = tool_data.get("score", 0)
        

        # Convert cosine(-1→1) to percentage(0→100)
        confidence = round(((score+1)/2)*100,2)

        # If score too low
        if confidence < CONFIDENCE_THRESHOLD:
            return {"messages":[{"role":"assistant",
                "content": f"⚠ Low confidence ({confidence}%). Try rephrasing your question."}]}

        # Step 2 — Generate final summarized answer using context
        final_prompt = f"""
        Answer the user's question using ONLY this context. Do not copy raw text.
        Summarize & explain clearly. If answer not found, say so.

        Question: {user_query}

        Context: {context}
        """

        answer = model.invoke(final_prompt).content.strip()

        final = f"""
📌 **Answer from documents:**

{answer}

**Confidence:** {confidence}%
"""
        return {"messages":[{"role":"assistant","content":final}]}

    except Exception as e:
        return {"messages":[{"role":"assistant","content":f"Error: {str(e)}"}]}
