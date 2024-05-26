import streamlit as st
from PyPDF2 import PdfReader
#is used to split large texts into smaller chunks for easier processing and analysis.
from langchain.text_splitter import RecursiveCharacterTextSplitter
#it's used to load environment variables.
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#or interacting with Google Generative AI models.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
#class is used to structure and represent text data.
from langchain_core.documents import Document
from dotenv import load_dotenv
import google.generativeai as genai
#it is used to access pre-trained models for tasks like summarization.
from transformers import pipeline

# Load environment variables
load_dotenv()
api_key = os.getenv("AIzaSyCYOfsGu4s6Qd304XNvt1pRFpaQFGo15HY")
genai.configure(api_key=api_key)

#This function extracts text from each page of the uploaded PDF documents.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#This function splits the extracted text into smaller chunks for easier processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

#This function tokenizes the text chunks and converts them into sequences of integers
def get_tokenizer_and_sequences(text_chunks):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_chunks)
    sequences = tokenizer.texts_to_sequences(text_chunks)
    word_index = tokenizer.word_index
    return tokenizer, sequences, word_index

#This function pads the sequences to ensure they are of equal length
def get_padded_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, max_length


def build_lstm_model(vocab_size, embedding_dim, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_model(model, padded_sequences):
    if len(padded_sequences) > 1:
        X_train, X_test = train_test_split(padded_sequences, test_size=0.2, random_state=42)
        y_train, y_test = np.expand_dims(X_train, -1), np.expand_dims(X_test, -1)  # Reshape for sparse_categorical_crossentropy
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    else:
        st.warning("Not enough data to split into train and test sets. Training skipped.")
    return model

def get_lstm_embeddings(model, padded_sequences):
    embeddings = model.predict(padded_sequences)
    return embeddings

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#This function summarizes the extracted text using a pre-trained BART model
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Split the text into chunks that are not too large
    max_chunk_size = 1024
    text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in text_chunks]
    return " ".join(summaries)

def user_input(user_question, embeddings, text_chunks):
    # Instead of loading from FAISS, we'll directly use the embeddings and text_chunks
    # Use some method to find the most relevant chunks based on the embeddings
    # Here we just mock the behavior with the first chunk
    docs = [Document(page_content=text_chunks[0])]

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("PDF Chat Summarizer")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        embeddings = np.load('lstm_embeddings.npy')  # Load precomputed embeddings
        text_chunks = st.session_state.get('text_chunks', [])
        if text_chunks:
            user_input(user_question, embeddings, text_chunks)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state['text_chunks'] = text_chunks

                tokenizer, sequences, word_index = get_tokenizer_and_sequences(text_chunks)
                padded_sequences, max_length = get_padded_sequences(sequences)
                
                vocab_size = len(word_index) + 1
                embedding_dim = 100

                lstm_model = build_lstm_model(vocab_size, embedding_dim, max_length)
                trained_model = train_lstm_model(lstm_model, padded_sequences)
                
                embeddings = get_lstm_embeddings(trained_model, padded_sequences)
                np.save('lstm_embeddings.npy', embeddings)  # Save embeddings for later use

                # Summarize the PDF content
                summary = summarize_text(raw_text)
                st.subheader("PDF Summary")
                st.write(summary)

                st.success("Done")

if __name__ == "__main__":
    main()
