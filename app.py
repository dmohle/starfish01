# project: starFish01
# RAG and streamlit
# dH 27dec23
# dependencies: pip install streamlit pypdf2 langchain python_dotenv faiss-cpu openai huggingface_hub
#   tiktoken InstructorEmbedding sentence_transformers

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from langchain.vectorstores import FAISS

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


def main():
    print("\n\n Welcome to Project StarFish01\n\n")
    st.set_page_config(page_title="Author with Multiple PDFs", page_icon=":books:")
    st.header("Author with Multiple PDFs :books:")
    st.text_input("Write a chapter from your PDFs...")

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
                # Use games and charles for testing...

                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                # use OpenAI Ada v2, this costs tokens
                # alternative is instructor-embeddings
                vectorstore = get_vectorstore(text_chunks)


if __name__ == "__main__":
    main()
