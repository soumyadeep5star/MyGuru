# MyGuru.AI

A multi-source AI knowledge assistant that searches Confluence documentation, GitHub repositories, and a structured database to answer questions. Built with Streamlit, Azure OpenAI GPT-4o, LangChain agents, and FAISS vector search.

## Features

🧠 **Intent-Routed Multi-Agent System**
- Supervisor agent uses GPT-4o to classify query intent before routing
- Routes queries only to the relevant agent (Confluence, GitHub, or Database)
- Falls back to Confluence when routing returns no usable result

✨ **Multi-Source Search**
- Confluence: searches all spaces and PDF attachments via semantic vector search
- GitHub: searches indexed README files from accessible repositories
- Database: executes natural-language-to-SQL queries against a structured database

🔍 **Semantic Chunking**
- Uses LangChain `SemanticChunker` with percentile-based breakpoints for Confluence content
- GitHub READMEs are stored as single chunks (one document, one vector)
- Falls back to `RecursiveCharacterTextSplitter` (chunk size 1200, overlap 150) if SemanticChunker is unavailable

📦 **FAISS Vector Store**
- Separate FAISS indexes for Confluence and GitHub content
- Chunking reports saved as JSON after each indexing run

🤖 **AI-Powered Responses**
- Azure OpenAI GPT-4o generates answers grounded strictly in retrieved content
- No hallucination from prior knowledge — agents only use tool-returned data
- Source links included in every response

🎨 **Streamlit UI**
- Clean dark-themed chat interface
- Fixed header, full-width messages, and persistent chat history per session

## Architecture

```
User Query
    │
    ▼
Supervisor Agent (GPT-4o intent router)
    │
    ├── Intent: "confluence"  →  Confluence Agent  →  FAISS (Confluence index)
    ├── Intent: "github"      →  GitHub Agent      →  FAISS (GitHub index)
    ├── Intent: "database"    →  Database Agent    →  SQLite / Azure SQL
    │
    └── Fallback: Confluence Agent (when routed agents return no useful result)
         │
         ▼
    Merge outputs  →  Return combined answer with source links
```

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with:
  - GPT-4o deployment
  - text-embedding-3-small deployment
- Confluence Cloud instance with API access
- GitHub personal access token (for GitHub indexing)

## Installation

1. **Clone the repository or navigate to the project directory**

```bash
cd MyGuru.AI-main
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Confluence
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your_email@example.com
CONFLUENCE_API_TOKEN=your_confluence_api_token

# GitHub (optional)
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_ORGANIZATION=your_org_name           # optional
GITHUB_INDEX_REPOS=org/repo1,org/repo2      # optional, comma-separated
```

### Getting a Confluence API Token

1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click **Create API token**, give it a label, and copy the token
3. Use your Atlassian email as `CONFLUENCE_USERNAME` and the token as `CONFLUENCE_API_TOKEN`

### Setting up Azure OpenAI

1. Create an Azure OpenAI resource in the Azure portal
2. Deploy `gpt-4o` and `text-embedding-3-small` models
3. Copy the endpoint URL and API key into `.env`

## Indexing

### Index Confluence content

```bash
python setup_index.py
```

This will:
1. Connect to all accessible Confluence spaces
2. Fetch every page and PDF attachment
3. Apply semantic chunking to each document
4. Generate embeddings and build a FAISS index
5. Save `vector_store/index.faiss` and `vector_store/confluence_chunking_report.json`

### Index GitHub READMEs

```bash
python setup_github_index.py
```

Or specify repositories explicitly:

```bash
python setup_github_index.py --repos org/repo1 org/repo2
```

This will:
1. Fetch the README from each repository (one chunk per README)
2. Generate embeddings and build a separate FAISS index
3. Save `vector_store/github_index.faiss` and `vector_store/github_chunking_report.json`

> Both indexing runs must complete before starting the app.

## Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## How It Works

### Query Flow

1. User submits a query via the chat interface
2. **Supervisor agent** sends the query to GPT-4o for intent classification
3. GPT-4o returns a JSON payload identifying the required data sources (e.g., `{"targets": ["confluence"]}`)
4. Only the selected agents are invoked
5. Each selected agent calls its tool, which searches the relevant FAISS index or database
6. Results from all invoked agents are merged
7. If no agent returns a useful answer, the Confluence agent is called as a fallback
8. Final response with source links is displayed in the chat

### Semantic Chunking

- Confluence documents are split using `SemanticChunker` (percentile threshold 85)
- If `langchain_experimental` is unavailable, `RecursiveCharacterTextSplitter` is used (size 1200, overlap 150)
- GitHub READMEs are stored as single chunks (`force_single_chunk=True`)
- Each chunk stores metadata: `doc_id`, `title`, `space`, `space_key`, `type`, `url`, `chunk_index`
- A JSON chunking report is saved after every indexing run

### Database Agent

- Converts natural language to SQL `SELECT` queries
- Currently uses a local SQLite in-memory database seeded with demo retail data:
  - `stores_operations` — 50 rows of daily store operation records
  - `retail_inventory_management` — 50 rows of product inventory records
- To connect to Azure SQL instead, configure `AZURE_SQL_SERVER`, `AZURE_SQL_DATABASE`, `AZURE_SQL_USERNAME`, and `AZURE_SQL_PASSWORD` in `.env` and update `database_search.py`

## Project Structure

```
MyGuru.AI-main/
├── app.py                      # Streamlit UI and app initialization
├── agents.py                   # Confluence, GitHub, Database, and Supervisor agents
├── agent_tools.py              # LangChain tool wrappers for each agent
├── config.py                   # Environment variable configuration
├── confluence_fetcher.py       # Confluence API integration
├── github_search.py            # GitHub API integration and document builder
├── vector_store.py             # FAISS vector store (index, save, load, search)
├── semantic_chunking.py        # SemanticChunkingService with fallback splitter
├── database_search.py          # SQLite / Azure SQL database searcher
├── sqlite_seed_data.py         # In-memory SQLite seed data (demo tables)
├── setup_index.py              # Confluence indexing script
├── setup_github_index.py       # GitHub indexing script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not committed)
├── vector_store/
│   ├── index.faiss             # Confluence FAISS index
│   ├── metadata.pkl            # Confluence documents + chunk report
│   ├── github_index.faiss      # GitHub FAISS index
│   ├── github_metadata.pkl     # GitHub documents + chunk report
│   ├── confluence_chunking_report.json
│   └── github_chunking_report.json
└── README.md
```

## Troubleshooting

**Import errors**
```bash
pip install -r requirements.txt
```

**Confluence index not found**
```bash
python setup_index.py
```

**GitHub index not found**
```bash
python setup_github_index.py
```

**Confluence connection issues**
- Verify `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, and `CONFLUENCE_API_TOKEN` in `.env`
- Ensure the API token is still valid

**Azure OpenAI errors**
- Verify `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`
- Confirm deployment names match `AZURE_OPENAI_CHAT_DEPLOYMENT` and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- Check that you have sufficient quota for GPT-4o and text-embedding-3-small

**GitHub token errors**
- Ensure `GITHUB_TOKEN` has `repo` and `read:org` scopes
- For private repositories, confirm the token has access

## Updating Indexes

To re-index after new content is added:

```bash
python setup_index.py          # Rebuild Confluence index
python setup_github_index.py   # Rebuild GitHub index
```

## Contributing

Feel free to submit issues or pull requests to improve the assistant.

## License

MIT License

