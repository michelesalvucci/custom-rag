#!/usr/bin/env python3
"""
RAG Index Creation Script using LlamaIndex with OpenAI
"""
import argparse
import glob
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create or update a RAG index from document sources."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="File, directory, or glob pattern to index (repeatable).",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[".md", ".java"],
        help="File extensions to include (repeatable). Default: .md .java",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns to exclude (repeatable).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "When updating an existing index, remove any previously indexed "
            "documents with the same file path before adding the new version."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    recursive = not args.no_recursive

    # Configure Hugging Face BGE-M3 embedding model
    print("Initializing BAAI/bge-m3 embedding model (Hugging Face)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    print("Loading documents...")
    
    # Collect all matching files based on CLI arguments
    documents = []
    seen_paths = set()
    base_path = Path(".")
    include_exts = [ext if ext.startswith(".") else f".{ext}" for ext in args.ext]

    def add_documents(docs, source_label):
        new_docs = []
        for doc in docs:
            file_path = doc.metadata.get("file_path")
            if file_path:
                normalized_path = str(Path(file_path).resolve())
                if normalized_path in seen_paths:
                    continue
                doc.metadata["file_path"] = normalized_path
                seen_paths.add(normalized_path)
            new_docs.append(doc)
        documents.extend(new_docs)
        print(f"  Loaded {len(new_docs)} documents from {source_label}")

    def expand_source(source):
        if any(char in source for char in "*?[]"):
            return glob.glob(source, recursive=recursive)
        return [source]

    for source in args.sources:
        matches = expand_source(source)
        if not matches:
            print(f"  Warning: No matches for {source}")
            continue

        for match in matches:
            path = (base_path / match).resolve()
            if path.is_dir():
                reader = SimpleDirectoryReader(
                    input_dir=str(path),
                    required_exts=include_exts,
                    recursive=recursive,
                    exclude=args.exclude,
                )
                try:
                    add_documents(reader.load_data(), str(path))
                except Exception as e:
                    print(f"  Warning: Could not load from {path}: {e}")
            elif path.is_file():
                reader = SimpleDirectoryReader(
                    input_files=[str(path)],
                    required_exts=include_exts,
                    exclude=args.exclude,
                )
                try:
                    add_documents(reader.load_data(), str(path))
                except Exception as e:
                    print(f"  Warning: Could not load from {path}: {e}")
            else:
                print(f"  Warning: Path does not exist: {path}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    if not documents:
        print("No documents found. Please check your arguments.")
        return

    # --- Chunking & metadata (replica di create_index_lang_chain.py) ---

    def _header_path(meta: dict) -> list[str]:
        keys = [f"Header {i}" for i in range(1, 7)]
        return [meta[k] for k in keys if k in meta and meta[k]]

    def _stable_id(source: str, path_parts: list[str]) -> str:
        import hashlib

        raw = f"{source}::{' > '.join(path_parts)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

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
        return_each_line=False,
    )

    hierarchical_sections = []
    for doc in documents:
        # Trattiamo il path del file come "source" per allinearci a LangChain
        source = doc.metadata.get("file_path", "unknown")
        md_splits = markdown_splitter.split_text(getattr(doc, "text", ""))

        for section in md_splits:
            path_parts = _header_path(section.metadata)
            parent_parts = path_parts[:-1]

            section_id = _stable_id(source, path_parts if path_parts else ["ROOT"])
            parent_section_id = _stable_id(source, parent_parts) if parent_parts else None

            section.metadata.update(
                {
                    "source": source,
                    "file_path": source,
                    "header_path": " > ".join(path_parts) if path_parts else "ROOT",
                    "header_level": len(path_parts),
                    "section_id": section_id,
                    "parent_section_id": parent_section_id,
                    "root_header": path_parts[0] if path_parts else None,
                }
            )

            hierarchical_sections.append(section)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
    )

    chunk_documents: list[Document] = []
    for section in hierarchical_sections:
        chunks = text_splitter.split_documents([section])
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            header_path = chunk.metadata.get("header_path", "ROOT")

            # Allineiamo il contenuto ai chunk LangChain
            page_content = (
                f"Source: {source}\n"
                f"Section: {header_path}\n\n"
                f"{chunk.page_content}"
            )

            chunk.metadata["chunk_in_section"] = i

            chunk_documents.append(
                Document(
                    text=page_content,
                    metadata=dict(chunk.metadata),
                )
            )

    print(f"Created {len(chunk_documents)} chunks from documents")

    # --- Creazione/persistenza indice FAISS ---
    storage_dir = Path(__file__).parent / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Dimensione embeddings per FAISS
    sample_emb = Settings.embed_model.get_text_embedding("test")
    dim = len(sample_emb)
    faiss_index = faiss.IndexFlatL2(dim)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("\nCreating new FAISS-backed vector index...")
    index = VectorStoreIndex.from_documents(
        chunk_documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # Persist index + FAISS index
    print("\nPersisting index to ./storage...")
    storage_context.persist(persist_dir=str(storage_dir))

    print("\n✓ RAG index created successfully!")
    print("  Index stored in: ./storage")
    print(f"  Indexed {len(chunk_documents)} chunks da {len(documents)} documenti sorgente")


if __name__ == "__main__":
    main()
