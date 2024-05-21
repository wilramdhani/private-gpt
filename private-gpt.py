from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
import os
from threading import Lock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import langid
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

lock = Lock()
conversation_chain = None

# Get the LLM URL from the environment variable
LLM_URL = os.getenv("LLM_URL")


def get_pdf_text(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits text into manageable chunks.

    Args:
        text (str): The text to be split.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Creates a vector store from text chunks.

    Args:
        text_chunks (list): The text chunks to be embedded and stored.

    Returns:
        FAISS: The vector store containing the text embeddings.
    """
    embeddings = OllamaEmbeddings(
        model="llama3",
        base_url="http://103.125.100.23"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def create_conversation_chain(vectorstore):
    """
    Creates a conversation chain with a language model and vector store.

    Args:
        vectorstore (FAISS): The vector store to be used for retrieval.

    Returns:
        ConversationalRetrievalChain: The initialized conversation chain.
    """
    llm = Ollama(
        model="llama3",
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        base_url="http://103.125.100.23",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|start_header_id|>", "<|end_header_id|>",
              "<|eot_id|>", "<|reserved_special_token"]
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def detect_language(text):
    """
    Detects the language of a given text.

    Args:
        text (str): The text whose language needs to be detected.

    Returns:
        dict: A dictionary containing the language and confidence score.
    """
    lang, confidence = langid.classify(text)
    language_names = {
        'en': 'English',
        'id': 'Indonesian',
    }
    language_full_name = language_names.get(lang, lang)
    result = {
        'language': language_full_name,
        'confidence': confidence
    }
    return result


def handle_userinput(user_question, conversation_chain):
    """
    Handles user input by passing the question to the conversation chain and retrieving the response.

    Args:
        user_question (str): The question asked by the user.
        conversation_chain (ConversationalRetrievalChain): The conversation chain to use for generating the response.

    Returns:
        str: The last AI response from the conversation chain.
    """
    response = conversation_chain.invoke({'question': user_question})
    chat_history = response['chat_history']

    # Find the last AI response
    for message in reversed(chat_history):
        if message.type == 'ai':
            return message.content


def main():
    """
    The main function that processes PDFs in a directory, sets up the conversation chain, and handles user questions.
    """
    pdf_directory = "source_data"  # Set your PDF directory here

    combined_text = ""

    # Process each PDF file in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            combined_text += get_pdf_text(pdf_path) + "\n"

    text_chunks = get_text_chunks(combined_text)

    global conversation_chain

    # Initialize the conversation chain if it hasn't been initialized yet
    with lock:
        if conversation_chain is None:
            vectorstore = get_vectorstore(text_chunks)
            conversation_chain = create_conversation_chain(vectorstore)

    # Handle user questions in a loop
    while True:
        user_question = input("Enter your question: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        response = handle_userinput(user_question, conversation_chain)
        print("\n --- \n")


if __name__ == '__main__':
    main()
