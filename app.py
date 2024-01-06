# project: starFish01
# RAG and streamlit
# dH 27dec23
# dependencies: pip install streamlit pypdf2 langchain python_dotenv faiss-cpu openai huggingface_hub
#   tiktoken InstructorEmbedding sentence_transformers
# updated 05 Jan 23 for flex day demo
# dH, Fresno, CA

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

load_dotenv()


def get_text_from_pdfs(pdf_docs):
    # this tiny variable will contain all the text of the pdfs
    text = ""
    for pdf in pdf_docs:
        # initialize one PdfReader() object for each pdf in our collection
        pdf_reader = PdfReader(pdf)
        # use the pdf_reader object to get text from the pdfs page by page
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        # chunk size is in characters
        # experiment with overlap for fine-tuning performance
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # try the non-fee instructor embedding
    # this takes forever!
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    print("\n\n Welcome to Project StarFish01\n\n")
    st.set_page_config(page_title="Author with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Author with Multiple PDFs :books:")
    user_question = st.text_input("Write a chapter from your PDFs...")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Robot!"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human !!!"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your source documents")
        # Combine all pdfs into one object named pdf_docs
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process PDFs'", accept_multiple_files=True)
        if st.button("Process PDFs"):
            # add a spinner
            with st.spinner("Processing multiple PDFs!"):
                # get text from pdfs

                raw_text = get_text_from_pdfs(pdf_docs)

                # st.write(raw_text)
                # error received 28 Dec regarding unicode character
                # Use games and charles OER stuff for Python testing...

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                # use OpenAI Ada v2, this costs tokens
                # alternative is instructor-embeddings
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
