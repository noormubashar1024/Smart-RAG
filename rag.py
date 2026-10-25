import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Configuration
st.set_page_config(
    page_title="Nexus PDF RAG Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pro Tech Styling & Animations
st.markdown("""
    <style>
                #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.2); }
        50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.4); }
        100% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.2); }
    }

    .stApp {
        animation: fadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Auth Card Styling */
    .auth-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 4rem;
    }

    .auth-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.08), 0 0 20px rgba(99, 102, 241, 0.05);
        animation: fadeIn 0.6s ease-out, pulseGlow 4s infinite;
        text-align: center;
        width: 100%;
        max-width: 480px;
    }

    .auth-card h2 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.75rem;
    }

    .auth-card p {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Pro Workspace Cards */
    .pro-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04);
        animation: fadeIn 0.5s ease-out;
        margin-top: 1.5rem;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State for Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# --- AUTHENTICATION GATE (Centered 20-30% Box) ---
if not st.session_state.authenticated:
    st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("""
            <div class="auth-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔐</div>
                <h2>Secure Access Gateway</h2>
                <p>Enter your OpenAI API key to initialize the high-precision vector engine and unlock your workspace.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        entered_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", label_visibility="collapsed")
        
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Unlock Workspace 🚀", use_container_width=True):
            if entered_key.strip().startswith("sk-") or len(entered_key.strip()) > 20:
                st.session_state.api_key = entered_key.strip()
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("⚠️ Please enter a valid OpenAI API key.")
    st.stop()

# --- MAIN APP (Unlocked Interface - API Key hidden from view) ---
os.environ["OPENAI_API_KEY"] = st.session_state.api_key

# Sidebar: Advanced Retrieval Settings (No API key options shown here)
with st.sidebar:
    st.markdown("### 🎛️ Neural Parameters")
    model_name = st.selectbox("Chat Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, step=50)
    top_k = st.slider("Retrieved Chunks (k)", 2, 10, 4)
    
    st.markdown("---")
    if st.button("🔒 Lock & Reset Key", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        st.rerun()

# App Header
st.title("⚡ Nexus PDF RAG Intelligence")
st.caption("Advanced document reasoning powered by high-performance vector retrieval and LLMs.")

# PDF Vectorstore Builder Function
def build_vectorstore(pdf_bytes: bytes, size: int, overlap: int) -> FAISS:
    """Load PDF, split into chunks, embed, and store in FAISS."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)

# File Uploader Interface
uploaded_pdf = st.file_uploader("📂 Upload Target Document (.PDF)", type=["pdf"])

if uploaded_pdf is not None:
    file_sig = (uploaded_pdf.name, uploaded_pdf.size, chunk_size, chunk_overlap)
    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("✨ Vectorizing document chunks and building neural embeddings..."):
            st.session_state.vectorstore = build_vectorstore(
                uploaded_pdf.getvalue(), chunk_size, chunk_overlap
            )
            st.session_state.file_sig = file_sig
            st.session_state.messages = []
            st.success(f"Successfully indexed **{uploaded_pdf.name}**! You can now query your document below.")

# RAG Chain Configuration
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert AI intelligence assistant that answers questions strictly using the "
     "provided context from a PDF document.\n"
     "Rules:\n"
     "1. Answer ONLY from the context below.\n"
     "2. If the answer is not in the context, say: \n"
     "\"I couldn't find that in the document.\"\n\n"
     "3. Cite the page number(s) you used, e.g. (p. 3).\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?') + 1}]\n{d.page_content}"
        for d in docs
    )

def get_chain(vectorstore: FAISS, model: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ), retriever

# Chat Interface Logic
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("💬 Query document contents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        chain, retriever = get_chain(
            st.session_state.vectorstore, model_name, top_k
        )

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing neural context and generating response..."):
                answer = chain.invoke(question)
                st.markdown(answer)

            # Transparency sources expander
            with st.expander("📌 Retrieved Context Chunks"):
                for doc in retriever.invoke(question):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1 if isinstance(page, int) else page}**")
                    st.text(doc.page_content[:500])
                    st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.markdown("""
        <div class="pro-card">
            <h3>📁 Workspace Initialized</h3>
            <p>Upload a PDF document above to activate semantic search and begin querying your data.</p>
        </div>
    """, unsafe_allow_html=True)
