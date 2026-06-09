import os
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    /* Global layout & responsiveness tuning */
    .main {
        padding-top: 2rem;
    }
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Elegant Header Branding */
    .header-container {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 1.5rem;
    }
    .header-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #FF8383, #1E90FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05rem;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #8c96a0;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Premium Assistant Response Container */
    .answer-box {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    /* Clean UI elements style adjustments */
    div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        background: transparent !important;
    }
    
    /* Custom Sidebar adjustments */
    .sidebar-desc {
        font-size: 0.95rem;
        color: #a3a8b4;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)




embedding_model = MistralAIEmbeddings()

llm = ChatMistralAI(
    model="mistral-small-latest"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Use ONLY the provided context.

            If the answer is not present in the context,
            say:
            "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)


with st.sidebar:
    st.markdown("<h2 style='font-weight: 700; margin-bottom: 0;'>DocuMind</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8c96a0; margin-bottom: 1.5rem;'>Intelligent Document Context RAG</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF context source",
        type=["pdf"],
        help="Upload your PDF file here to let the model index it dynamically."
    )

    st.markdown("---")
    
    st.markdown("### 💡 Quick Guide")
    st.markdown(
        """
        <div class="sidebar-desc">
        1. Drop your PDF file into the upload block above.<br>
        2. Wait for the pipeline processing checkmark.<br>
        3. Drop a contextual text prompt in the entry console to chat.
        </div>
        """, 
        unsafe_allow_html=True
    )


if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    
    with st.sidebar.status("🔄 Processing & indexing PDF...", expanded=True) as status:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )
        status.update(label=f"{uploaded_file.name} ready!", state="complete", expanded=False)



if os.path.exists("chroma_db"):

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    st.markdown("""
    <div class="header-container">
        <div class="header-title">Book RAG Assistant</div>
        <div class="header-subtitle">Your document is indexed. Query insights effortlessly below.</div>
    </div>
    """, unsafe_allow_html=True)

   
    query = st.chat_input("Ask anything about your document...")

    if query:
        
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing context blocks..."):
                docs = retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )

                response = llm.invoke(final_prompt)

        
            st.markdown(
                f"""
                <div class="answer-box">
                    {response.content}
                </div>
                """,
                unsafe_allow_html=True
            )

            
            with st.expander("📄 View Retrieved Source Context Chunks"):
                st.text_area("Context Block", value=context, height=250, disabled=True)

else:
    
    st.markdown("""
    <div class="header-container" style="margin-top: 10%;">
        <div class="header-title">Book RAG Assistant</div>
        <div class="header-subtitle">Chat with your PDF documents effortlessly using modern AI pipelines.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.info(" To begin querying your documents, please drop a PDF file inside the Sidebar workspace area.")