import streamlit as st
import tiktoken
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.vectorstores.annoy import Annoy
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import os
import shutil
import time
from txtfile import append_text_to_file
import ollama
from chromadb.utils import embedding_functions
from langchain.embeddings import SentenceTransformerEmbeddings
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
previous_question = None
system_message_prompt = SystemMessagePromptTemplate.from_template(
    """ {context} You are a QA bot. As a QA bot, your primary responsibility is to thoroughly absorb and comprehend information shared by the user. You need to understand the audience and their queries' intent.
                In the event that the user poses questions aligned with the information previously provided, respond in a professional, detailed, and unambiguous manner. The customer should not get confused with navigation and should not seek for human intervention. You must Emphasize the importance of understanding the user's intention and what they are seeking to ensure accurate and comprehensive responses."""
)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def count_tokens_pdf(chunks):
    tokens = encoding.encode(chunks)
    return len(tokens)


def get_vectorstore(text_chunks):
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings,persist_directory=".\\Chroma")
    
    return vectorstore


def get_savedvectorstore():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=".\\Chroma", embedding_function=embeddings)
    return vectordb


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = ChatOllama(model="mistral", temperature=0, streaming=True)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                system_message_prompt,
                human_message_prompt,
            ]),
        }
    )
    return conversation_chain


def handle_userinput(user_question):
    global previous_question
    if user_question.startswith("#info"):
        updated_info = user_question[len("#info"):].strip()
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        if st.session_state.chat_history:
            previous_question = st.session_state.chat_history[-4].content
        append_text_to_file(output_file_path='./Docs/output_file.txt', text_to_append=updated_info, question_to_append=previous_question, max_size_kb=1024)
    else:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        previous_question = user_question

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot",
                       page_icon=":robot_face:")
    
    st.write(css, unsafe_allow_html=True)
    st.markdown("""
      <style>

      .block-container
      {
        padding-top: 1rem;
        
        margin-top: 1rem;
      }

    </style>
    """, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        with st.spinner("Loading conversation database"):
            vectorstore = get_savedvectorstore()
            st.session_state.conversation = get_conversation_chain(vectorstore)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    styl = f"""
          <style>
        
          .stchatInput {{
            position: fixed;
            bottom: 1.5rem;
            z-index: 3; /* Set z-index higher than overlay bar to bring it to the front */
          }}
          </style>
          """
    user_question = st.chat_input(placeholder="Ask anything...", key=None)  # Add a placeholder text

    if user_question:
        handle_userinput(user_question)
        st.markdown("""
                  <script>
                  var element = document.querySelector(".chat-message.bot");
                  element.scrollIntoView({ behavior: 'smooth', block: 'end' });
                  </script>
                  """, unsafe_allow_html=True)
    st.markdown(styl, unsafe_allow_html=True)




    with st.sidebar:
        st.header("Chat with Us!")
        st.markdown("""
    <style>
                    
    .big-font {
        font-size:24px !important;
        color: #45A046 !important; /* Light green */
        font-family: 'Museo Sans', sans-serif !important; /* Museo Sans 700 */
        font-weight: 700 !important; /* Bold */
        margin-top: -60px; /* Example: Add 20px margin above text */
    }
    .small-font {
        font-size:14px !important;
        color: #FFFFFF !important; /* White */
        font-family: 'Museo Sans', sans-serif !important; /* Museo Sans 700 */
        
        text-align: justify
    }

""", unsafe_allow_html=True)
        

        st.markdown('<p class="big-font">Chat with Us!</p></div>', unsafe_allow_html=True)
        st.markdown('<p class="small-font">Welcome! I\'m your friendly virtual assistant here to assist you with any queries or issues you may have regarding our services. If I happen to make any mistakes along the way, please don\'t hesitate to let me know. I\'m constantly learning and improving to provide you with the best assistance possible. Let\'s get started in resolving your concerns together!</p></div>', unsafe_allow_html=True)

        Commented out since it's for clearing data
        with st.spinner("Clearing Data"):
            for root, dirs, files in os.walk('.\\Chroma'):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

        Commented out since it's for PDF processing
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'.", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                st.write(count_tokens_pdf(raw_text))
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
        if st.button("Chat with persisted DB"):
            with st.spinner("Loading"):
                vectorstore = get_savedvectorstore()
                st.session_state.conversation = get_conversation_chain(vectorstore)
    

if __name__ == '__main__':
    main()



