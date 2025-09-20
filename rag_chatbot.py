import sys
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

# Core libraries for RAG implementation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class RAGChatbot:
  
    
    def __init__(self, openai_api_key: str, use_free_models: bool = False):
       
        self.openai_api_key = openai_api_key
        self.use_free_models = use_free_models
        self.vectorstore = None
        self.qa_chain = None
        
        # Set up embeddings (OpenAI or HuggingFace)
        if use_free_models:
            print("⚠️  Using free HuggingFace models (slower but free)")
            # Uncomment below and install sentence-transformers for free embeddings
            # self.embeddings = HuggingFaceEmbeddings(
            #     model_name="all-MiniLM-L6-v2"
            # )
            raise NotImplementedError("Free models setup - see comments in code")
        else:
            print("🚀 Using OpenAI models")
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.embeddings = OpenAIEmbeddings()
        
        # Set up the language model
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # Fast and cost-effective
            temperature=0,  # Deterministic responses
            max_tokens=300  # Keep responses concise
        )
        
        print("✅ RAG Chatbot initialized successfully!")
    
    def load_documents(self, file_path: str) -> List[Document]:
      
        print(f"📂 Loading documents from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")
        
        documents = []
        
        if os.path.isdir(file_path):
            # Load all supported files from directory
            loader = DirectoryLoader(
                file_path,
                glob="**/*.txt",  # You can add "**/*.pdf" for PDFs
                loader_cls=TextLoader
            )
            documents = loader.load()
        elif file_path.endswith('.txt'):
            # Load single text file
            loader = TextLoader(file_path)
            documents = loader.load()
        elif file_path.endswith('.pdf'):
            # Load single PDF file
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        print(f"✅ Loaded {len(documents)} document(s)")
        return documents
    
    def process_documents(self, documents: List[Document]) -> None:
        """
        Process documents: split into chunks, create embeddings, store in vector database.
        
        Args:
            documents (List[Document]): List of documents to process
        """
        print("⚙️  Processing documents...")
        
        # Step 1: Split documents into smaller chunks
        # This helps with more precise retrieval and fits LLM context limits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Size of each chunk (characters)
            chunk_overlap=200,      # Overlap between chunks (preserves context)
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Split into {len(chunks)} chunks")
        
        # Step 2: Create embeddings and store in FAISS vector database
        print("🔮 Creating embeddings and building vector database...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        print("✅ Documents processed and stored in vector database!")
    
    def setup_qa_chain(self) -> None:
        """
        Set up the Question-Answering chain with custom prompt.
        This ensures the bot only answers from provided context.
        """
        # Custom system prompt - CRITICAL for RAG behavior
        prompt_template = """You are a helpful assistant that answers questions ONLY based on the provided context below. 

IMPORTANT RULES:
1. Answer ONLY using information from the context provided
2. If the answer cannot be found in the context, respond with: "I could not find this in the documents provided."
3. Keep responses short, clear, and factual
4. Do not use external knowledge or make assumptions
5. Quote relevant parts from the context when possible

Context:
{context}

Question: {question}

Answer:"""
        
        # Create the prompt template
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Set up the QA chain with our custom prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Simple approach: stuff all retrieved docs into prompt
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False  # Set to True if you want to see sources
        )
        
        print("🔗 Question-answering chain set up successfully!")
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question to the RAG chatbot.
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The chatbot's response
        """
        if not self.qa_chain:
            return "❌ Chatbot not ready. Please load and process documents first."
        
        try:
            # Get response from the QA chain
            response = self.qa_chain.invoke({"query": question})
            return response["result"]
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"
    
    def save_vectorstore(self, save_path: str) -> None:
        """Save the vector database to disk for future use."""
        if self.vectorstore:
            self.vectorstore.save_local(save_path)
            print(f"💾 Vector database saved to: {save_path}")
    
    def load_vectorstore(self, load_path: str) -> None:
        """Load a previously saved vector database."""
        if os.path.exists(load_path):
            self.vectorstore = FAISS.load_local(load_path, self.embeddings)
            print(f"📁 Vector database loaded from: {load_path}")
        else:
            raise FileNotFoundError(f"Vector database not found: {load_path}")


def create_sample_document():
    """Create a sample document for testing the chatbot."""
    sample_content = """
# Python Programming Guide

Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability and simplicity.

## Key Features of Python
- Easy to learn and use
- Interpreted language (no compilation needed)
- Object-oriented programming support
- Large standard library
- Cross-platform compatibility
- Dynamic typing

## Python Applications
Python is widely used for:
1. Web development (Django, Flask)
2. Data science and machine learning
3. Automation and scripting
4. Game development
5. Desktop applications

## Python Syntax Basics
Variables in Python don't need explicit declaration:
```
name = "Python"
version = 3.9
```

Python uses indentation to define code blocks, making it very readable.

## Data Types
Python has several built-in data types:
- int: Integer numbers
- float: Decimal numbers  
- str: Text strings
- list: Ordered collections
- dict: Key-value pairs
- bool: True/False values

## Popular Libraries
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib: Data visualization
- Requests: HTTP library
- TensorFlow: Machine learning

Python's philosophy is summarized in "The Zen of Python": Beautiful is better than ugly, explicit is better than implicit, simple is better than complex.
"""
    
    with open("sample.txt", "w") as f:
        f.write(sample_content)
    print(" 1) Created sample.txt with Python programming content")


def main():
    """Main function to run the RAG chatbot CLI interface."""
    
    print("=" * 60)
    print(" RAG CHATBOT - Document Q&A System")
    print("=" * 60)
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OpenAI API key not found!")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr enter it now:")
        api_key = input("OpenAI API Key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            sys.exit(1)
    
    try:
        # Initialize the chatbot
        chatbot = RAGChatbot(api_key)
        
        # Check if sample document exists, create if not
        if not os.path.exists("sample.txt"):
            create_sample_document()
        
        # Load and process documents
        print("\n" + "="*40)
        documents = chatbot.load_documents("sample.txt")
        chatbot.process_documents(documents)
        chatbot.setup_qa_chain()
        
        print("\n🎉 Chatbot is ready! You can now ask questions about the documents.")
        print("📋 Try asking about Python features, applications, or syntax.")
        print("💡 Type 'quit' or 'exit' to stop the chatbot.")
        print("="*40)
        
        # Main chat loop
        while True:
            print("\n" + "-"*40)
            question = input("❓ Your Question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for using the RAG chatbot!")
                break
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            # Get answer from chatbot
            print("🤔 Thinking...")
            answer = chatbot.ask_question(question)
            print(f"🤖 Answer: {answer}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the RAG chatbot!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your API key and internet connection.")


if __name__ == "__main__":

    main()
