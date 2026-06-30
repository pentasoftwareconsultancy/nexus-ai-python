import os
import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==================================================
# LOAD ENVIRONMENT
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("Nexus CTS Chatbot")
st.write("Ask questions from the PDF document.")

# ==================================================
# PDF CONFIGURATION
# ==================================================

PDF_NAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "uploads",
    "company_policy.pdf"
)


def get_pdf_path(pdf_name):
    """
    Returns absolute PDF path from data folder.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(
        base_dir,
        "data",
        pdf_name
    )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    return pdf_path


# ==================================================
# EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==================================================
# VECTOR STORE CREATION
# ==================================================

@st.cache_resource
def create_vectorstore(pdf_path):

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = load_embedding_model()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore, len(documents), len(chunks)


# ==================================================
# LOAD PDF ON STARTUP
# ==================================================

try:

    pdf_path = get_pdf_path(PDF_NAME)

    if "vectorstore" not in st.session_state:

        with st.spinner("Loading PDF and creating vector database..."):

            vectorstore, pages, chunks = create_vectorstore(
                pdf_path
            )

            st.session_state.vectorstore = vectorstore

            st.success(
                f"PDF Loaded Successfully | Pages: {pages} | Chunks: {chunks}"
            )

except Exception as e:

    st.error(str(e))
    st.stop()

# ==================================================
# CHAT HISTORY
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==================================================
# QUESTION INPUT
# ==================================================

query = st.text_input(
    "Ask a question"
)

if query:

    with st.spinner("Searching document..."):

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 20
            }
        )

        docs = retriever.invoke(query)

        context = "\n\n".join([
            f"[SOURCE {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

    chat_history = "\n".join([
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in st.session_state.chat_history[-5:]
    ])

    prompt = f"""
You are Nexus CTS AI Assistant.

ROLE:
You are a professional document question-answering assistant.

Your primary job is to answer ONLY using the provided PDF context.

==================================================
RULES
==================================================

1. Use PDF information whenever available.

2. Never invent facts.

3. Never claim information exists in the document if it does not.

4. If information is partially available:
   - Use document information first.
   - Clearly mention what comes from the document.

5. If information is not found:

Return EXACTLY:

"I couldn't find information related to your query in the available knowledge base. For further assistance, please contact Nexus Corporate Center at +91-9545450788 or +91-9545450677 or email nexusCTC2020@gmail.com."

6. If the question is unclear:
   Ask a clarification question.

==================================================
RESPONSE FORMAT RULES

ALWAYS prefer structured output.

A. If answer contains multiple items:

Return:

## Answer

- Item 1
- Item 2
- Item 3

--------------------------------------------------

B. If answer contains steps:

Return:

## Steps

1. Step One
2. Step Two
3. Step Three

--------------------------------------------------

C. If answer contains comparison:

Return a markdown table.

Example:

| Feature | Description |
|----------|------------|
| A | Value |
| B | Value |

--------------------------------------------------

D. If answer contains:

- Courses
- Benefits
- Policies
- Rules
- Departments
- Features
- Requirements
- Services
- Products

Return bullet list format.

--------------------------------------------------

E. For simple factual questions:

Return:

## Answer

<short answer>

Maximum 3 lines.

--------------------------------------------------

F. Never write long paragraphs.

Maximum paragraph size:
2 lines.

G. Never repeat the same information multiple times.

==================================================
CONTEXT
==================================================

{context}

==================================================
CHAT HISTORY
==================================================

{chat_history}

==================================================
QUESTION
==================================================

{query}

==================================================
ANSWER
==================================================
"""

    with st.spinner("Generating Answer..."):

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            top_p=0.9,
            messages=[
                {
                    "role": "system",
                    "content": "You are a PDF document QA assistant. Always return structured answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content
        answer = answer.strip()

        while "\n\n\n" in answer:
            answer = answer.replace("\n\n\n", "\n\n")

    st.session_state.chat_history.append({
        "question": query,
        "answer": answer
    })

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(docs, start=1):

            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)
            st.divider()


# ==================================================
# FASTAPI APP
# ==================================================

app = FastAPI(
    title="Nexus CTS RAG API",
    description="PDF RAG Chatbot API for frontend integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ==================================================
# API STATE (shared in-memory store for API calls)
# ==================================================

api_state = {
    "vectorstore": None,
    "chat_history": []
}


@app.on_event("startup")
def startup_load_pdf():
    vectorstore, pages, chunks = create_vectorstore(PDF_NAME)
    api_state["vectorstore"] = vectorstore
    print(f"API: PDF Loaded | Pages: {pages} | Chunks: {chunks}")


# ==================================================
# SCHEMAS
# ==================================================

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


class HistoryItem(BaseModel):
    question: str
    answer: str


# ==================================================
# POST /ask
# ==================================================

@app.post("/ask", response_model=AnswerResponse)
def ask_question(body: QuestionRequest):

    if not api_state["vectorstore"]:
        raise HTTPException(status_code=503, detail="Vector store not ready yet.")

    query = body.question.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = api_state["vectorstore"].as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20
        }
    )

    docs = retriever.invoke(query)

    context = "\n\n".join([
        f"[SOURCE {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    chat_history = "\n".join([
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in api_state["chat_history"][-5:]
    ])

    prompt = f"""
You are Nexus CTS AI Assistant.

ROLE:
You are a professional document question-answering assistant.

Your primary job is to answer ONLY using the provided PDF context.

==================================================
RULES
==================================================

1. Use PDF information whenever available.

2. Never invent facts.

3. Never claim information exists in the document if it does not.

4. If information is partially available:
   - Use document information first.
   - Clearly mention what comes from the document.

5. If information is not found:

Return EXACTLY:

"I couldn't find information related to your query in the available knowledge base. For further assistance, please contact Nexus Corporate Center at +91-9545450788 or +91-9545450677 or email nexusCTC2020@gmail.com."

6. If the question is unclear:
   Ask a clarification question.

==================================================
RESPONSE FORMAT RULES

ALWAYS prefer structured output.

A. If answer contains multiple items:

Return:

## Answer

- Item 1
- Item 2
- Item 3

--------------------------------------------------

B. If answer contains steps:

Return:

## Steps

1. Step One
2. Step Two
3. Step Three

--------------------------------------------------

C. If answer contains comparison:

Return a markdown table.

Example:

| Feature | Description |
|----------|------------|
| A | Value |
| B | Value |

--------------------------------------------------

D. If answer contains:

- Courses
- Benefits
- Policies
- Rules
- Departments
- Features
- Requirements
- Services
- Products

Return bullet list format.

--------------------------------------------------

E. For simple factual questions:

Return:

## Answer

<short answer>

Maximum 3 lines.

--------------------------------------------------

F. Never write long paragraphs.

Maximum paragraph size:
2 lines.

G. Never repeat the same information multiple times.

==================================================
CONTEXT
==================================================

{context}

==================================================
CHAT HISTORY
==================================================

{chat_history}

==================================================
QUESTION
==================================================

{query}

==================================================
ANSWER
==================================================
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": "You are a PDF document QA assistant. Always return structured answers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content
    answer = answer.strip()

    while "\n\n\n" in answer:
        answer = answer.replace("\n\n\n", "\n\n")

    api_state["chat_history"].append({
        "question": query,
        "answer": answer
    })

    return AnswerResponse(question=query, answer=answer)


# ==================================================
# GET /history
# ==================================================

@app.get("/history", response_model=list[HistoryItem])
def get_history():
    return api_state["chat_history"]