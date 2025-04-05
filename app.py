import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load API keys
load_dotenv("key.env")
groq_api_key = os.getenv("chatbot_api_key")
langchain_api_key = os.getenv("langsmith_api_key")

# LangChain environment setup
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Groq_ChatBot"

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# LangGraph state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Chatbot logic
def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

# Build LangGraph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Streamlit UI
st.set_page_config(page_title="🧠 Chat with AI", layout="centered")
st.title("🧠 Chat with AI (LangGraph + Groq)")
st.markdown("A fast & intelligent chatbot powered by **LangGraph** and **Groq LLM**.")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Type your message here...")

# Only run graph if user submitted a message
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        try:
            for event in graph.stream({"messages": st.session_state.chat_history}):
                for value in event.values():
                    response = value["messages"]
                    st.session_state.chat_history.append(("assistant", response.content))
        except Exception as e:
            st.error(f"Error: {e}")

# Display conversation
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)
