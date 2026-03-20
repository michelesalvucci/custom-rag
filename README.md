# Custom RAG

This project implements a Retrieval-Augmented Generation (RAG) system for an R&D initiative within my organization. It enables semantic search and question-answering over proprietary documentation, with all sensitive client information anonymized to protect confidentiality.

## Key Points
- Semantic search uses embeddings from OpenAI's `text-embedding-3-small` model
- LLM responses powered by OpenAI chat models (default: `gpt-4o-mini`)
- Vector storage uses a local FAISS index (`faiss_index/`)
- Requires an OpenAI API key to function

## Quick Start

### Prerequisites
1. Obtain an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   **Tip**: Add this to your `~/.bashrc` or `~/.zshrc` to persist across sessions

### Installation
1. Create and activate the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Create and Query the Index
1. Create the index (from one or more files/directories):
   ```bash
   python3 create_index_lang_chain.py project_docs/
   ```

   Multiple sources and custom output path:
   ```bash
   python3 create_index_lang_chain.py project_docs/ samples/auth-example -o faiss_index
   ```
   
2. Query the index (single question):
   ```bash
   python3 query_index_lang_chain.py "controller and services responsible for joining rules and initiatives"
   ```

3. Interactive chat mode (maintains conversation context):
   ```bash
   python3 query_index_lang_chain.py
   ```
   
   In interactive mode:
   - Ask follow-up questions without repeating context
   - Type `reset` to clear conversation history
   - Type `exit`, `quit`, or `q` to quit

## Configuration
- Adjust model settings in `create_index_lang_chain.py` and `query_index_lang_chain.py`:
  - Change embedding model (e.g., `text-embedding-3-large`)
   - Change LLM model with `--model` (e.g., `gpt-4.1`, `gpt-4o-mini`)
   - Adjust retrieval depth with `-k/--top-k`
   - Adjust generation behavior with `--temperature`

Useful query options:
```bash
python3 query_index_lang_chain.py -i faiss_index -k 5 --model gpt-4.1 --temperature 0.0 "your question"
```

### Temperature (`--temperature`)
`temperature` controls the level of variability (randomness) in the model's response generation:
- `0.0`: more deterministic and consistent output (preferred for technical/documentation Q&A)
- `0.2 - 0.5`: balance between accuracy and flexibility
- `> 0.7`: more creative and less predictable output

In practice, it does not change document retrieval from the vector index, but only **how** the model formulates the final response based on the retrieved context.

## How does it work

### Document Loading
Documents are loaded from files or folders using LangChain loaders:
- `DirectoryLoader` for directories
- `TextLoader` for single files

This allows indexing mixed sources in one run.

### Chunking (LangChain)
Chunking is the process of breaking down documents into smaller, coherent pieces that can be efficiently processed by language models. This is essential because LLMs cannot process entire projects at once.

It reduces input size for LLM processing, allows passing only relevant document sections, and improves response accuracy while reducing API costs.

This project uses a two-step split strategy:
1. `MarkdownHeaderTextSplitter` to split by Markdown sections (`#` to `######`)
2. `RecursiveCharacterTextSplitter` for final chunk sizing (`chunk_size=700`, `chunk_overlap=150`)

```python
markdown_splitter = MarkdownHeaderTextSplitter(
   headers_to_split_on=[
     ("#", "Header 1"),
     ("##", "Header 2"),
     ("###", "Header 3"),
     ("####", "Header 4"),
     ("#####", "Header 5"),
     ("######", "Header 6"),
   ],
      return_each_line=False
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
```

### Section Hierarchy Preservation
Hierarchy between sections is preserved during indexing.

Instead of concatenating all documents and splitting once, each source document is split independently. This prevents losing file boundaries and keeps subsection relationships consistent within the same file.

For each markdown section, the indexer stores hierarchy metadata:
- `header_path`: full path like `Auth > OAuth > Refresh Token`
- `header_level`: depth in the hierarchy (1..6)
- `section_id`: stable identifier of the current section
- `parent_section_id`: stable identifier of the direct parent section (or `None` for top-level)
- `root_header`: first-level section name
- `source`: original file path

When a section is further split into character chunks, each chunk keeps the same section metadata plus:
- `chunk_in_section`: chunk order inside the same section

This allows retrieval logic to link a subsection back to its father section (via `parent_section_id`) and to group sibling chunks by section (`section_id`).

### Embedding (OpenAI)
Embedding is the transformation of text or code into numerical vectors that represent semantic meaning. Each chunk is converted into a high-dimensional vector space where semantically similar content clusters together.

**Example:** "AuthService handles login" → [0.021, -0.44, 0.98, ...]

**Implementation:** OpenAI Embedding API (`text-embedding-3-small` model in this project)

### Vector Index (FAISS via LangChain)
A Vector Index is an optimized data structure that stores embeddings and enables rapid similarity searches. It is not an AI system but rather an efficient indexing mechanism for quick retrieval.

It stores all document embeddings for fast access, enables millisecond-level search across hundreds or thousands of chunks and replaces inefficient one-to-one chunk comparison.

**Implementation:** LangChain `FAISS.from_documents(...)`, persisted locally with `save_local("faiss_index")`.

```python
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
```

### Semantic Retrieval (LangChain Retriever)
Semantic retrieval finds the most semantically similar chunks to a given query, rather than relying on keyword matching. It works even when query uses different terminology than source code.

**Process:**
1. Convert user query to embedding using OpenAI
2. Compare query embedding against indexed embeddings
3. Return top-K most similar chunks

**Implementation:** FAISS retriever with configurable top-K (`--top-k`, default `3`).
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
```

### Conversation Memory
Interactive mode keeps an in-memory history of previous user/assistant turns and injects recent turns into the prompt, allowing for:
- Follow-up questions without repeating context
- Reference to previous answers
- Multi-turn conversations about the codebase

Example conversation (real case):
```
You: dove è implementata angrafica gestori?
Assistant:  L'anagrafica gestori è implementata nel controller "OutletManagersController" e nel servizio "CustomerService"s

You: che microservizio è?
Assistant: Si trova nel microservizio "outlets".
```

### Language Model (OpenAI)
The final component: a generative language model that synthesizes retrieved chunks into coherent, natural language responses. The LLM reads contextual chunks and generates explanations, reasoning, and answers.

It synthesizes and explains retrieved information, provides natural language responses to user queries and applies reasoning across multiple chunks.

**Implementation:** `ChatOpenAI` with configurable `--model` and `--temperature`.
