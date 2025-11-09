import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import AgentType, initialize_agent, agent_types
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name = "Search")

st.title(" Langchain - Chat with search")
"""
In this example we're using SteamlitcallbackHandler to display the thoughts & actions """



# side-bar for settings :-
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type = "password")

if "messages"not in st.session_state:
  st.session_state["messages"] = [
    {"role": "Assistant", "content" : "Hi, I am a Chatbot who can search the Web. How can i help you?"}
  ]


for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"]) 


if prompt:= st.chat_input(placeholder="What is Machine Learning?"):
  st.session_state.messages.append({"role" : "user", "content" : prompt})
  st.chat_message("user").write(prompt)

  llm = ChatGroq(api_key = api_key, model_name = "openai/gpt-oss-120b", streaming= True)
  tools = [search, arxiv, wikipedia]

  search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True)

  with st.chat_message("Assistant"):
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
    response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
    st.session_state.messages.append({'role' : 'Assistant', "content" : response})
    st.write(response)    
