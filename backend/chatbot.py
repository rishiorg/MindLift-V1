from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Setup device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# Initialize embedding model
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Groq LLM
try:
    groq_api_key = os.environ.get("GROQ_API_KEY", "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")
    if not groq_api_key:
        print("Warning: GROQ_API_KEY not found in environment variables. Using fallback responses.")
    
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=1024
    )
except Exception as e:
    print(f"Error initializing Groq LLM: {e}")
    llm = None

# Create conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create a mental health knowledge base
try:
    # First check if we have a persisted vector store
    if os.path.exists("./chroma_db"):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        # Load mental health resources
        # Note: You should create a 'knowledge' directory with relevant PDF files
        if os.path.exists("./"):
            loader = DirectoryLoader(
                "./",
                glob="Psychology The Science of Mind and Behaviour (Richard Gross) (Z-Library).pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            vectorstore.persist()
        else:
            print("Knowledge directory not found. Creating empty vector store.")
            vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
            vectorstore.persist()
except Exception as e:
    print(f"Error creating vector store: {e}")
    # Create a simple in-memory vector store as fallback
    vectorstore = None

# Create conversation chain
if llm and vectorstore:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True
    )
else:
    qa_chain = None

def get_chatbot_response(user_message):
    """
    Get response from the chatbot.
    
    Args:
        user_message (str): User's message
        
    Returns:
        str: Chatbot response
    """
    try:
        if qa_chain:
            # Use the QA chain to get a response
            result = qa_chain({"question": user_message})
            return result["answer"]
        else:
            # Fallback responses when LLM is not available
            import random
            fallback_responses = [
                "I'm here to listen. Could you tell me more about how you're feeling?",
                "That sounds challenging. How have you been coping with this situation?",
                "I understand this might be difficult. What support do you have available?",
                "It's important to take care of your mental health. Have you considered speaking with a professional?",
                "Thank you for sharing that with me. What would help you feel better right now?",
                "I'm sorry to hear you're going through this. Remember that it's okay to ask for help.",
                "Let's focus on one thing at a time. What's your biggest concern right now?",
                "Deep breathing can help in stressful moments. Would you like to try a quick breathing exercise?",
                "Your feelings are valid. Is there something specific you'd like guidance on?",
                "Sometimes writing down our thoughts can help clarify them. Have you tried journaling?"
            ]
            return random.choice(fallback_responses)
    except Exception as e:
        print(f"Error getting chatbot response: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Could you try again in a moment?"

# Create a knowledge directory if it doesn't exist
if not os.path.exists("./knowledge"):
    os.makedirs("./knowledge")
    print("Created knowledge directory. Please add PDF files with mental health information.")

# Create a test response to ensure everything is working
if __name__ == "__main__":
    test_response = get_chatbot_response("Hello, how are you?")
    print(f"Test response: {test_response}")