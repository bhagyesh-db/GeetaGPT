import streamlit as st
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
global hf
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

global retriever
global new_db
new_db = FAISS.load_local("faiss_index", hf)
retriever = new_db.as_retriever(search_kwargs={"k":5})

global url 
global header
url = 'https://5000-01hx6yx4rzcwy9z7j41pznp5jr-w.cloudspaces.litng.ai'
headers = {'Content-Type': 'application/json'}


def faiss_search(query):
  """
  Performs text similarity search using FAISS.

  Args:
      query: The user's query string.
      faiss_index: The loaded FAISS index for efficient retrieval.
      hf: The Hugging Face library for embedding generation.

  Returns:
      A list of top K similar document IDs based on the query.
  """
  context=retriever.get_relevant_documents(query)
  return context

def llm_query(context, question):
    prompt_template = f"""Final Prompt: Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, strictly don't add anything from your side.

    Context: {context}
    Question: {question}

    Only return the helpful answer. give direct answer with reference from the context
    answer:
    """
    data_to_llm = {'text_corpus': prompt_template}
    llm_json_data = json.dumps(data_to_llm)
    response = requests.post(url, headers=headers, data=llm_json_data)
    return response.json()
 
def chat(user_input):
  """
  Processes user input and retrieves relevant documents using FAISS.

  Args:
      user_input: The user's query string.
      documents: The processed and prepared documents.

  Returns:
      A response string based on the retrieved documents.
  """
  # response 
  retrieved_docs = faiss_search(user_input)
  context_page_content = [doc.page_content for doc in retrieved_docs]
  context_page_content_as_text = '\n'.join(context_page_content)

  # RAG logic 
  final_output = llm_query(context_page_content_as_text, user_input)
  return final_output

if "message_history" not in st.session_state:
  st.session_state["message_history"] = []

message_history = st.session_state["message_history"]

user_input = st.chat_input(placeholder="Ask question Krishna will answer your questions!!!")

if user_input:
  message_history.append({"role": "You", "content": user_input})
  response = chat(user_input)
  message_history.append({"role": "Krishna", "content": response})

st.subheader("Bhagwat Geeta 🦚")
for message in message_history:
  st.write(f"{message['role']}: {message['content']}")


# to run this streamlit application use following command
# streamlit run app.py
