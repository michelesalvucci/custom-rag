from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv
import os
import argparse
import glob
from pathlib import Path
import hashlib

load_dotenv()

def _header_path(meta: dict) -> list[str]:
  keys = [f"Header {i}" for i in range(1, 7)]
  return [meta[k] for k in keys if k in meta and meta[k]]

def _stable_id(source: str, path_parts: list[str]) -> str:
  raw = f"{source}::{' > '.join(path_parts)}"
  return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def create_index(sources, index_path: str = "faiss_index"):
  """
  Create a FAISS index from documents using LangChain and OpenAI.
  
  Args:
    sources: List of file paths or directories to index
    index_path: Path to save the FAISS index
  """
  
  # Initialize OpenAI embeddings
  embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
  )
  
  # Load documents from sources
  all_documents = []
  
  for source in sources:
    source_path = Path(source)
    
    if source_path.is_dir():
      # Load all files from directory
      loader = DirectoryLoader(str(source_path), glob="**/*", loader_cls=TextLoader, show_progress=True)
      documents = loader.load()
      all_documents.extend(documents)
      print(f"Loaded {len(documents)} documents from directory: {source}")
    elif source_path.is_file():
      # Load single file
      loader = TextLoader(str(source_path))
      documents = loader.load()
      all_documents.extend(documents)
      print(f"Loaded file: {source}")
    else:
      print(f"Warning: {source} is not a valid file or directory, skipping...")
  
  if not all_documents:
    raise ValueError("No documents were loaded. Please provide valid files or directories.")
  
  print(f"\nTotal documents loaded: {len(all_documents)}")
  
  # Split documents into chunks based on markdown headers
  headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
  ]
  
  markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_line=False
  )
  
  # Split by markdown headers per document and keep hierarchy metadata
  hierarchical_sections = []
  for doc in all_documents:
    source = doc.metadata.get("source", "unknown")
    md_splits = markdown_splitter.split_text(doc.page_content)

    for section in md_splits:
      path_parts = _header_path(section.metadata)
      parent_parts = path_parts[:-1]

      section_id = _stable_id(source, path_parts if path_parts else ["ROOT"])
      parent_section_id = _stable_id(source, parent_parts) if parent_parts else None

      section.metadata.update({
        "source": source,
        "header_path": " > ".join(path_parts) if path_parts else "ROOT",
        "header_level": len(path_parts),
        "section_id": section_id,
        "parent_section_id": parent_section_id,
        "root_header": path_parts[0] if path_parts else None,
      })

      print(f"Section: {section.metadata}")

      hierarchical_sections.append(section)
  
  # Further split large sections if needed
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
  )

  docs = []
  for section in hierarchical_sections:
    chunks = text_splitter.split_documents([section])
    for i, chunk in enumerate(chunks):
      source = chunk.metadata.get("source", "unknown")
      header_path = chunk.metadata.get("header_path", "ROOT")
      chunk.page_content = (
        f"Source: {source}\n"
        f"Section: {header_path}\n\n"
        f"{chunk.page_content}"
      )
      chunk.metadata["chunk_in_section"] = i
      docs.append(chunk)
  
  print(f"Created {len(docs)} chunks from documents")
  
  # Create FAISS index
  vectorstore = FAISS.from_documents(docs, embeddings)
  
  # Save index
  vectorstore.save_local(index_path)
  print(f"Index created and saved to {index_path}")
  
  return vectorstore

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Create a FAISS index from documents using LangChain and OpenAI",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Index a single file
  python create_index_lang_chain.py file1.md
  
  # Index multiple files
  python create_index_lang_chain.py file1.md file2.md file3.md
  
  # Index a directory
  python create_index_lang_chain.py ./documents/
  
  # Index files with custom output path
  python create_index_lang_chain.py file1.md file2.md -o my_custom_index
  
  # Mix files and directories
  python create_index_lang_chain.py ./project_docs/ ./samples/auth-example
    """
  )
  
  parser.add_argument(
    'sources',
    nargs='+',
    help='Files or directories to index (can specify multiple)'
  )
  
  parser.add_argument(
    '-o', '--output',
    default='faiss_index',
    help='Path to save the FAISS index (default: faiss_index)'
  )
  
  args = parser.parse_args()
  
  # Create index from provided sources
  create_index(args.sources, args.output)