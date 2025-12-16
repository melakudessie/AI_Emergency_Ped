"""
Clinical Guidelines Assistant - Railway Optimized
RAG-based PDF Query System with Streamlit
"""

import streamlit as st
import os
from typing import List, Dict
from io import BytesIO
import tempfile

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="Clinical Guidelines Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class RAGSystem:
    """Handles all RAG operations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = None
        self.vectorstore = None
        self.chain = None
        
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        """Load embedding model with caching - Railway optimized"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for Railway
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def process_pdf(self, pdf_file) -> List:
        """Process PDF file and return documents"""
        try:
            # Create temporary file (Railway-compatible)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Split documents - optimized for Railway memory limits
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks for better memory management
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            splits = text_splitter.split_documents(documents)
            return splits
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
    
    def create_vectorstore(self, documents: List):
        """Create FAISS vectorstore from documents"""
        try:
            if not self.embeddings:
                self.embeddings = self.load_embedding_model()
            
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            return True
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return False
    
    def create_chain(self):
        """Create conversational retrieval chain"""
        try:
            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=1024,
                streaming=True
            )
            
            # Create custom prompt
            prompt_template = """You are an expert medical assistant analyzing clinical guidelines and WHO documents. 
            Use the following pieces of context to answer the question at the end. 
            
            If you don't know the answer based on the context, say "I cannot find this information in the provided document."
            Always cite the page number when referencing information.
            
            Context: {context}
            
            Question: {question}
            
            Provide a detailed, accurate answer with page citations:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3, "fetch_k": 5}  # Reduced for Railway
                ),
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                verbose=False
            )
            
            return True
        except Exception as e:
            st.error(f"Error creating chain: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict:
        """Query the RAG system"""
        try:
            if not self.chain:
                return {"error": "System not initialized. Please process a PDF first."}
            
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            return {"error": f"Query error: {str(e)}"}


def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'doc_name' not in st.session_state:
        st.session_state.doc_name = None


def sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.markdown("## 🏥 Clinical Guidelines Assistant")
        st.markdown("---")
        
        # API Key input - Railway uses environment variables
        api_key = os.environ.get("GROQ_API_KEY", "")
        
        if not api_key:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key. Get one at console.groq.com"
            )
        else:
            st.success("✅ API Key loaded from environment")
        
        # PDF Upload
        st.markdown("### 📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload WHO reports, clinical guidelines, or medical documents"
        )
        
        # Process button
        if uploaded_file and api_key:
            if st.button("🚀 Process PDF", use_container_width=True):
                process_pdf_file(uploaded_file, api_key)
        
        # Settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Higher values make output more creative"
            )
            
            st.info("💡 Railway Optimizations Active:\n- Memory-efficient chunking\n- CPU-based embeddings\n- Optimized vector search")
        
        # Info
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Features:**
        - 📚 PDF Analysis
        - 🔍 Smart Search
        - 💬 Conversational AI
        - 📊 Citation Tracking
        
        **Optimized for Railway:**
        - Low memory footprint
        - Fast response times
        - Ephemeral storage handling
        """)
        
        # Status
        if st.session_state.pdf_processed:
            st.success(f"✅ Document Ready: {st.session_state.doc_name}")
        
        return api_key


def process_pdf_file(uploaded_file, api_key: str):
    """Process uploaded PDF file"""
    with st.spinner("🔄 Processing PDF... This may take 30-60 seconds..."):
        try:
            # Initialize RAG system
            rag_system = RAGSystem(api_key)
            
            # Process PDF
            progress_bar = st.progress(0)
            st.info("📖 Loading PDF...")
            progress_bar.progress(25)
            
            documents = rag_system.process_pdf(uploaded_file)
            
            if not documents:
                st.error("Failed to process PDF. Please try again.")
                return
            
            st.info(f"✂️ Split into {len(documents)} chunks...")
            progress_bar.progress(50)
            
            # Create vectorstore
            st.info("🧠 Creating embeddings...")
            if not rag_system.create_vectorstore(documents):
                return
            
            progress_bar.progress(75)
            
            # Create chain
            st.info("🔗 Building query system...")
            if not rag_system.create_chain():
                return
            
            progress_bar.progress(100)
            
            # Save to session state
            st.session_state.rag_system = rag_system
            st.session_state.pdf_processed = True
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chat_history = []
            
            st.success("✅ PDF processed successfully! You can now ask questions.")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def display_chat_interface():
    """Display main chat interface"""
    st.markdown("<h1 class='main-header'>💬 Ask Questions About Your Document</h1>", 
                unsafe_allow_html=True)
    
    if not st.session_state.pdf_processed:
        st.info("👈 Please upload and process a PDF document from the sidebar to begin.")
        
        # Quick start guide
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 1️⃣ Upload")
            st.markdown("Select a PDF document from the sidebar")
        with col2:
            st.markdown("### 2️⃣ Process")
            st.markdown("Click 'Process PDF' and wait ~30 seconds")
        with col3:
            st.markdown("### 3️⃣ Query")
            st.markdown("Ask questions about your document")
        
        return
    
    # Quick action buttons
    st.markdown("### 🎯 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Summarize Document"):
            query_rag("Provide a comprehensive summary of this document including key points and main findings.")
    
    with col2:
        if st.button("🎯 Key Points"):
            query_rag("Extract and list the main key points and recommendations from this document.")
    
    with col3:
        if st.button("📊 Statistics"):
            query_rag("List all important statistics, numbers, and data mentioned in this document.")
    
    with col4:
        if st.button("✅ Recommendations"):
            query_rag("What are the main recommendations or action items in this document?")
    
    st.markdown("---")
    
    # Chat history display
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (Page {source['page']}):")
                        st.text(source['content'][:300] + "...")
    
    # Chat input
    if user_input := st.chat_input("Ask a question about the document..."):
        query_rag(user_input)


def query_rag(question: str):
    """Query the RAG system and display results"""
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.rag_system.query(question)
            
            if "error" in response:
                st.error(response["error"])
                return
            
            answer = response.get("answer", "No answer generated.")
            source_documents = response.get("source_documents", [])
            
            # Display answer
            st.markdown(answer)
            
            # Prepare sources
            sources = []
            if source_documents:
                for doc in source_documents:
                    sources.append({
                        "page": doc.metadata.get("page", "Unknown"),
                        "content": doc.page_content
                    })
                
                # Display sources
                with st.expander(f"📚 View {len(sources)} Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}** (Page {source['page']}):")
                        st.text(source['content'][:300] + "...")
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })


def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    api_key = sidebar()
    
    # Display main interface
    display_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🏥 Clinical Guidelines Assistant | Powered by Groq + LangChain | "
        "Deployed on Railway"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
