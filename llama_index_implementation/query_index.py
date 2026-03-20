#!/usr/bin/env python3
"""Query the RAG Index using LlamaIndex

Embeddings: BAAI/bge-m3 (Hugging Face)
Vector store: FAISS
Reranker: BAAI/bge-reranker-large
LLM: Llama 3.1 8B via Ollama
"""
import os
import sys
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore


load_dotenv()


def main():
    # Configure Hugging Face BGE-M3 embedding model (deve corrispondere a create_index.py)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # Configure Llama 3.1 8B via Ollama (assume modello locale "llama3.1:8b")
    Settings.llm = Ollama(
        model="llama3.1:8b",
        request_timeout=300.0,
        temperature=0.1,
    )

    # Configure reranker BGE (SentenceTransformerRerank usa sentence-transformers sotto)
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large",
        top_n=3,
    )

    # Load the FAISS-backed index
    print("Loading FAISS index from ./storage...")
    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage",
        vector_store=vector_store,
    )
    index = load_index_from_storage(storage_context)
    
    # Create chat memory to store conversation history
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    # Create chat engine with memory
    """
        Command with a single query would have been: index.as_query_engine()
    """
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        similarity_top_k=10,  # candidati prima del rerank
        max_tokens=3000,
        system_prompt=(
            "You are a helpful AI assistant that answers questions about the codebase. "
            "Use the provided context and conversation history to give accurate answers."
        ),
        node_postprocessors=[reranker],
    )
    
    if len(sys.argv) > 1:
        # Single query from command line
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}")
        print("\nGenerating response...")
        response = chat_engine.chat(query)
        print(f"\nResponse:\n{response}")
        
        # Show source nodes for debugging
        print("\n--- Retrieved Sources ---")
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\nSource {i}:")
            print(f"  File: {node.node.metadata.get('file_name', 'unknown')}")
            print(f"  Score: {node.score:.4f}")
            print(f"  Content preview: {node.node.text[:200]}...")
    else:
        # Interactive chat mode with context
        print("\nRAG Chat Interface (type 'exit' to quit, 'reset' to clear history)")
        print("The chat maintains context from previous questions.\n")
        
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            if query.lower() == "reset":
                memory.reset()
                print("✓ Conversation history cleared.")
                continue
                
            if query:
                print("\nAssistant: ", end="", flush=True)
                response = chat_engine.chat(query)
                print(f"{response}\n")


if __name__ == "__main__":
    main()
