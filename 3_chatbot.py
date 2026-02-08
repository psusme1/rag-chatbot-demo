import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("This project requires Python 3.12")

import os
from dotenv import load_dotenv
import streamlit as st

from pydantic import BaseModel
from typing import Any, Optional, Dict, Union
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)


###############################   INITIALIZE CHAT MODEL   #######################################################################################################
llm = ChatOllama(
    model=os.getenv("CHAT_MODEL"),
    temperature=0
)

# pulling prompt from hub
prompt = PromptTemplate.from_template("""
You are a helpful assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store and provide a response.

Use the tool 'retrieve' to get relevant information when the question is about the scraped content.

IMPORTANT:
- Do NOT output URLs.
- If you use retrieved info, cite sources as [1], [2], etc. (just the bracket numbers).
- The UI will display clickable links separately.

Query:
{input}

Chat history:
{chat_history}

Scratchpad:
{agent_scratchpad}
""")

# creating the retriever tool
from typing import Any, Union

from pydantic import BaseModel
from typing import Any, Optional, Dict, Union
from langchain_core.tools import tool

class RetrieveArgs(BaseModel):
    # allow absolutely anything; we'll normalize inside
    query: Optional[Union[str, Dict[str, Any]]] = None

@tool(args_schema=RetrieveArgs)
def retrieve(query: Optional[Union[str, Dict[str, Any]]] = None) -> str:
    """Retrieve information related to a query."""
    # Normalize tool inputs:
    # - query can be a string
    # - or a dict like {"object": "..."} or {"text": "..."} or even {}
    if isinstance(query, dict):
        query_text = (
            query.get("query")
            or query.get("object")
            or query.get("text")
            or query.get("input")
            or ""
        )
    else:
        query_text = query or ""

    # If the tool got called with no usable query, return empty
    if not query_text.strip():
        return ""

    retrieved_docs = vector_store.similarity_search(query_text, k=2)

    serialized = ""
    for doc in retrieved_docs:
        serialized += (
            f"Source: {doc.metadata.get('source','')}\n"
            f"Title: {doc.metadata.get('title','')}\n"
            f"Content: {doc.page_content}\n\n"
        )

    return serialized

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("How are you?")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    # Pull sources out of the tool output so the UI can display them as links
    sources = []
    for action, observation in result.get("intermediate_steps", []):
        if getattr(action, "tool", None) == "retrieve" and isinstance(observation, str):
            # Your retrieve tool returns blocks like:
            # Source: <url>\nContent: ...
            for block in observation.split("\n\n"):
                for line in block.splitlines():
                    if line.startswith("Source: "):
                        url = line.replace("Source: ", "").strip()
                        if url and url not in sources:
                            sources.append(url)

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        if sources:
            st.markdown("**Sources:**")
            for i, url in enumerate(sources, start=1):
                st.markdown(f"[{i}] {url}")


