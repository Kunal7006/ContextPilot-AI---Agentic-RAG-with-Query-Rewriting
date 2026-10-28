
# 🚀 ContextPilot AI — Agentic RAG with Gemini & Qdrant

**An intelligent, evaluation-driven Agentic Retrieval-Augmented Generation (RAG) system powered by Google Gemini, Agno, Qdrant Cloud, and LangSmith.**

ContextPilot AI enables users to interact with PDF documents and web content through natural-language queries. It combines **semantic vector retrieval, AI-powered query rewriting, grounded answer generation, and optional web-search fallback** to deliver context-aware responses.

The project also includes a **LangSmith-based evaluation framework** for benchmarking answer quality, retrieval relevance, hallucination resistance, and response latency.

## 📊 Performance Highlights

Evaluated using a **40-question benchmark** derived from a technical research thesis.

| Metric | Result |
|---|---|
| **Semantic Answer Correctness** | **91.0%** |
| **Faithfulness** | **99.0%** |
| **Context Relevance** | **93.25%** |
| **Average Response Latency** | **10.63 seconds** |
| **Evaluation Dataset** | **40 questions** |
| **Retrieval Configuration** | **Top-K = 5** |

> **Note:** Answer correctness, faithfulness, and context relevance are average **Gemini LLM-as-a-Judge scores**, not independently verified accuracy percentages. These results correspond to the **query-rewriting-disabled, Top-K=5** configuration.

## ✨ Features

### 📄 Document Processing

- **PDF Upload & Processing:** Upload PDF documents and extract their textual content.
- **Web Content Ingestion:** Extract and process content from web pages.
- **Intelligent Text Chunking:** Split documents into manageable overlapping chunks.
- **Vector Embeddings:** Generate **768-dimensional Gemini embeddings** for semantic search.
- **Qdrant Cloud Storage:** Store document embeddings and metadata in a cloud-based vector database.

### 🧠 Agentic Query Processing

- **Intelligent Query Rewriting:** Reformulate user queries using a Gemini-powered Agno agent to improve retrieval relevance.
- **Semantic Retrieval:** Retrieve the most relevant document chunks using Qdrant similarity search.
- **Configurable Top-K Search:** Experiment with different retrieval depths, including **K=3, K=5, and K=10**.
- **Context-Aware Answer Generation:** Generate responses grounded in retrieved document content.
- **Unsupported-Answer Handling:** Instruct the agent to acknowledge when retrieved context is insufficient.

### 🌐 Web Search Integration

- **Exa AI Integration:** Supports external web searches when additional information is required.
- **Web Search Fallback:** Provides an optional alternative when document retrieval cannot sufficiently answer a question.
- **Domain Filtering:** Supports restricting web searches to selected domains.
- **Source Attribution:** Displays supporting sources where available.

### 💬 Interactive Streamlit Interface

- **Natural-Language Chat:** Ask questions about uploaded documents.
- **PDF and URL Inputs:** Process multiple types of knowledge sources.
- **Chat History Management:** Maintain and clear conversation history.
- **Retrieval Transparency:** Inspect retrieved context and rewritten queries where exposed by the interface.
- **API Configuration:** Configure required credentials securely.

### 📈 LangSmith Observability & Evaluation

- **End-to-End Tracing:** Track query rewriting, document retrieval, and answer generation.
- **Faithfulness Evaluation:** Assess whether generated claims are supported by retrieved context.
- **Semantic Answer Correctness:** Compare generated answers with reference answers using Gemini as an evaluator.
- **Context Relevance:** Evaluate how well retrieved chunks address the original question.
- **Reference Token F1:** Measure lexical overlap between generated and reference answers.
- **Latency Tracking:** Record response time for each evaluation question.
- **Ablation Experiments:** Compare query rewriting enabled versus disabled.
- **Retrieval Experiments:** Compare retrieval performance across different Top-K configurations.

## 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │    Streamlit UI      │
                  │   User Interaction   │
                  └──────────┬───────────┘
                             │
                ┌────────────▼────────────┐
                │      User Query         │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │ Gemini Query Rewriter   │
                │    (Agno Agent)         │
                │       Optional          │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │  Gemini Query Embedding │
                │      768 Dimensions     │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │      Qdrant Cloud       │
                │    Vector Similarity    │
                │        Search           │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │   Retrieved Context     │
                │     Top-K Chunks        │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │   Gemini RAG Agent      │
                │  Grounded Generation    │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │    Final AI Response    │
                └─────────────────────────┘

         Optional: Exa AI Web Search Fallback

         LangSmith traces the RAG execution stages
         and evaluates response quality and latency.
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Programming Language** | Python |
| **Frontend** | Streamlit |
| **LLM** | Google Gemini |
| **Agent Framework** | Agno |
| **Vector Database** | Qdrant Cloud |
| **Embeddings** | Gemini Embedding 001 |
| **Retrieval Framework** | LangChain |
| **Document Processing** | PyPDFLoader, BeautifulSoup |
| **Web Search** | Exa AI |
| **Observability** | LangSmith |
| **Evaluation** | Gemini LLM-as-a-Judge |
| **Environment Management** | python-dotenv |

## 🔄 RAG Workflow

### Step 1: Document Ingestion

Users upload PDF documents or provide web URLs.

The system extracts text, splits it into overlapping chunks, and generates **768-dimensional vector embeddings** using Google's Gemini embedding model.

These embeddings are stored in **Qdrant Cloud**.

### Step 2: Query Processing

When a user submits a question, the system can optionally use a **Gemini-powered query rewriting agent** to reformulate the question for improved semantic retrieval.

### Step 3: Semantic Retrieval

The query is converted into a vector embedding and compared with stored document vectors in Qdrant.

The **Top-K most similar chunks** are retrieved and assembled into context.

### Step 4: Answer Generation

The Gemini RAG agent receives the original question, rewritten query, and retrieved context.

It generates a response grounded in the retrieved evidence.

### Step 5: Web Search Fallback

When enabled, **Exa AI** provides an additional search capability for questions requiring external information.

### Step 6: Observability

**LangSmith** captures execution traces, enabling inspection of retrieval behavior, generated responses, and pipeline latency.

## 📊 Experimental Evaluation

The system was evaluated using a **40-question dataset** based on a technical research thesis.

The benchmark includes questions about system architecture, methodology, algorithms, implementation details, and experimental results.

### Evaluation Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Measures whether generated factual claims are supported by retrieved documents |
| **Answer Correctness** | Evaluates semantic agreement with reference answers |
| **Context Relevance** | Measures whether retrieved chunks contain useful information for the question |
| **Reference Token F1** | Measures word-level overlap with reference answers |
| **Latency** | Measures end-to-end RAG response time |

The three semantic quality metrics were scored using **Gemini as an LLM-as-a-Judge**, with scores ranging from 0 to 1.

### Experiment 1: Query Rewriting ON vs OFF

| Metric | Rewriting OFF | Rewriting ON |
|---|---:|---:|
| **Faithfulness** | 99.0% | 99.5% |
| **Answer Correctness** | 91.0% | 90.5% |
| **Context Relevance** | 93.25% | 97.0% |
| **Average Latency** | 10.63 s | 19.48 s |
| **Top-K** | 5 | 5 |

**Key Findings:**

- Query rewriting improved **context relevance by 3.75 percentage points**.
- Faithfulness increased by **0.5 percentage points**.
- Semantic answer correctness decreased slightly by **0.5 percentage points**.
- Average latency increased from **10.63 seconds to 19.48 seconds**, reflecting the additional query-rewriting operation.

**Conclusion:** Query rewriting improved retrieval relevance but introduced additional latency without improving overall answer correctness in this benchmark.

### Experiment 2: Top-K Retrieval Comparison

| Metric | K=3 | K=5 |
|---|---:|---:|
| **Faithfulness** | 97.25% | 99.0% |
| **Answer Correctness** | 86.38% | 91.0% |
| **Context Relevance** | 92.75% | 93.25% |
| **Average Latency** | 9.86 s | 10.63 s |

**Key Findings:**

- Increasing retrieval depth from **K=3 to K=5** improved answer correctness by approximately **4.62 percentage points**.
- Faithfulness improved by **1.75 percentage points**.
- Average response latency increased by only **0.77 seconds**.

**Conclusion:** The **K=5 configuration** delivered better answer quality than K=3 with a modest latency increase.

### Recommended Configuration

Based on the measured experiments:

| Parameter | Recommended Value |
|---|---|
| **Embedding Model** | gemini-embedding-001 |
| **Embedding Dimensions** | 768 |
| **Vector Database** | Qdrant Cloud |
| **Retrieval Strategy** | Similarity Search |
| **Top-K** | 5 |
| **Query Rewriting** | Disabled by default for lower latency |
| **Answer Generation** | Gemini via Agno |
| **Evaluation Platform** | LangSmith |

> These findings are specific to the evaluated thesis dataset. Results may differ across documents, questions, models, and retrieval configurations.

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.10+**
- **Google AI Studio API Key**
- **Qdrant Cloud Cluster**
- **LangSmith API Key** for tracing and evaluation
- **Exa AI API Key** (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/contextpilot-ai-agentic-rag.git
cd contextpilot-ai-agentic-rag
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_google_api_key

QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Agentic-RAG_Gemini

EXA_API_KEY=your_exa_api_key
```

**Important:** Never commit `.env`, credentials, or API keys to GitHub.

### 5. Run the Application

```bash
streamlit run agentic_rag_gemini.py
```

Open the local Streamlit URL displayed in your terminal.

## 🧪 Running Evaluations

The evaluation scripts use the existing **LangSmith dataset** and **Qdrant collection**.

### Baseline Evaluation

```bash
python evaluate_rag.py
```

### Quality Evaluation

```bash
python evaluate_rag_v2.py
```

This measures:

- Reference Token F1
- Average response latency
- Faithfulness
- Semantic answer correctness
- Context relevance

### Additional Experiments

```bash
python run_extra_evaluations.py --suite all
```

Run individual experiments:

```bash
python run_extra_evaluations.py --suite rewrite
python run_extra_evaluations.py --suite topk
python run_extra_evaluations.py --suite robustness
```

**Note:** Evaluations require a populated Qdrant collection, the LangSmith evaluation dataset, valid API credentials, and sufficient Gemini API quota.

## 📁 Project Structure

```text
contextpilot-ai-agentic-rag/
│
├── agentic_rag_gemini.py
├── evaluate_rag.py
├── evaluate_rag_v2.py
├── run_extra_evaluations.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── evaluation_results/
    └── evaluation_results_summary.csv
```

The evaluation-results directory contains the summarized benchmark results.

## 🔐 Security & Best Practices

- Store credentials in **environment variables**.
- Exclude `.env` and `.venv` from Git tracking.
- Avoid uploading confidential PDFs or proprietary documents.
- Use **LangSmith tracing** to investigate retrieval and generation failures.
- Validate answers against retrieved evidence.
- Review LLM-generated evaluation scores before interpreting them as objective accuracy measurements.

## 🔮 Future Improvements

- **Hybrid Retrieval:** Combine keyword search and vector similarity.
- **Reranking:** Introduce cross-encoder reranking for retrieved chunks.
- **Retrieval Recall@K:** Evaluate against manually annotated relevant evidence.
- **Adaptive Retrieval:** Dynamically select retrieval depth based on question complexity.
- **Caching:** Reduce repeated embedding and generation calls.
- **Human Evaluation:** Compare LLM judge scores with manually reviewed answers.
- **Deployment:** Deploy the Streamlit application for public demonstration.

## 🎯 Project Highlights

- Built an **end-to-end Agentic RAG pipeline** integrating Gemini, Agno, Qdrant Cloud, and Streamlit.
- Implemented **PDF/web ingestion, query rewriting, semantic retrieval, and optional Exa AI search**.
- Developed a **40-question LangSmith evaluation benchmark** with five quality and performance metrics.
- Achieved **91% average LLM-judged answer correctness** and **99% faithfulness** in the K=5 baseline.
- Improved **retrieval context relevance by 3.75 percentage points** using query rewriting.
- Demonstrated measurable **retrieval-depth and latency trade-offs** through controlled experiments.

---

**ContextPilot AI — Building observable, measurable, and context-grounded AI systems.**
