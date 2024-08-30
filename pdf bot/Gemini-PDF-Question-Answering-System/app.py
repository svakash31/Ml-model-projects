import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()

# Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        if not pdf_reader.pages:
            continue
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    if text.strip() == "":
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faisss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()

def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "answer is not available in the context."
        
        Context:
        {context}?
        
        Question:
        {question}
        
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        st.stop()

def generate_mcqs(text, num_questions=1):
    sentences = [sentence.strip() for sentence in text.split('. ') if len(sentence.split()) > 3]
    mcqs = []

    for _ in range(num_questions):
        if not sentences:
            break
        question = random.choice(sentences)
        correct_answer = question
        sentences.remove(question)
        distractors = random.sample(sentences, min(3, len(sentences)))
        options = [correct_answer] + distractors
        random.shuffle(options)
        formatted_options = [
            f"A) {options[0]}",
            f"B) {options[1]}",
            f"C) {options[2]}",
            f"D) {options[3]}"
        ]
        mcqs.append({
            "question": f"Q: {question}?",
            "options": formatted_options,
            "answer": correct_answer
        })

    return mcqs

def display_mcqs(mcqs):
    for idx, mcq in enumerate(mcqs):
        st.markdown(f"**Q{idx + 1}: {mcq['question']}**")
        for option in mcq['options']:
            st.markdown(f"{option}")
        st.markdown(f"**Answer:** {mcq['answer']}")
        st.markdown("---")

def user_input(user_question, processed_pdf_text):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faisss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            st.write("No similar documents found.")
            return

        chain = get_conversational_chain()

        context = f"{processed_pdf_text}\n\nQuestion: {user_question}"

        response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
        response_text = response["output_text"]
        st.write(response_text)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"user": user_question, "bot": response_text})

    except Exception as e:
        st.error(f"Error during user input processing: {e}")

def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat by uploading PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a Question from the PDF Files or request MCQs")

    if user_question:
        if "generate mcqs" in user_question.lower():
            if st.session_state.get("processed_pdf_text"):
                num_questions = st.slider("Select number of MCQs", 1, 5, 1)
                mcqs = generate_mcqs(st.session_state["processed_pdf_text"], num_questions)
                display_mcqs(mcqs)
            else:
                st.error("Please upload PDF files first.")
        else:
            if st.session_state.get("pdf_docs"):
                processed_pdf_text = get_pdf_text(st.session_state["pdf_docs"])
                user_input(user_question, processed_pdf_text)
            else:
                st.error("Please upload PDF files first.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Files & Click Submit to Proceed", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = ""
                    text_chunks = []
                    for pdf in pdf_docs:
                        pdf_reader = PdfReader(pdf)
                        if not pdf_reader.pages:
                            continue
                        for page in pdf_reader.pages:
                            raw_text += page.extract_text()
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        vector_store = get_vector_store(text_chunks)
                        chain = get_conversational_chain()
                        st.session_state["pdf_docs"] = pdf_docs
                        st.session_state["text_chunks"] = text_chunks
                        st.session_state["vector_store"] = vector_store
                        st.session_state["processed_pdf_text"] = raw_text
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("No valid text found in the PDFs.")
            else:
                st.error("Please upload PDF files.")

        if st.button("Reset"):
            st.session_state.clear()
            st.experimental_rerun()

        if st.session_state.get("pdf_docs"):
            st.subheader("Uploaded Files:")
            for i, pdf_doc in enumerate(st.session_state["pdf_docs"]):
                st.write(f"{i+1}. {pdf_doc.name}")

    st.sidebar.title("Chat History")
    for chat in st.session_state.chat_history:
        st.sidebar.markdown(f"**You:** {chat['user']}")
        st.sidebar.markdown(f"**Bot:** {chat['bot']}")
        st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
