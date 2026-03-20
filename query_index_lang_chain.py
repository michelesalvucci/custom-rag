#!/usr/bin/env python3
"""
Query a LangChain FAISS index using OpenAI with optional interactive chat mode.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ModuleNotFoundError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()


def build_chain(index_path: str, api_key: str, model: str, temperature: float, top_k: int):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=300,
        max_tokens=900,
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, "
                "rewrite the question so it is a standalone query for retrieval. "
                "Do not answer the question.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant that answers questions about the indexed documents.
                Use only the provided context and chat history when possible.
                If you are unsure, say you do not know.\n\n
                Context:\n{context}""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_prompt = PromptTemplate.from_template(
        "Source: {source}\n"
        "Section: {header_path}\n"
        "Content:\n{page_content}"
    )

    qa_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def run_chat_mode(chain):
    history: list[HumanMessage | AIMessage] = []

    print("\nRAG Chat Interface (type 'exit' to quit, 'reset' to clear history)")
    print("The chat maintains context from previous questions.\n")

    while True:
        query = input("\n==================================================================\n>: ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            break

        if query.lower() == "reset":
            history.clear()
            print("✓ Conversation history cleared.")
            continue

        if not query:
            continue

        response = chain.invoke({"input": query, "chat_history": history[-16:]})
        answer = response.get("answer", "")

        print(f"\n---------------------------------------------------------------------\n{answer}")
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))


def main():
    parser = argparse.ArgumentParser(
        description="Query a LangChain FAISS index with optional chat mode",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--index-path",
        default="faiss_index",
        help="Path to the saved FAISS index directory (default: faiss_index)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=6,
        help="Number of retrieved chunks per query (default: 6)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI chat model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY='your-api-key'"
        )

    if not Path(args.index_path).exists():
        raise FileNotFoundError(
            f"Index path '{args.index_path}' not found. "
            "Create it first using create_index_lang_chain.py"
        )

    print(f"Loading FAISS index from ./{args.index_path}...")
    chain = build_chain(
        index_path=args.index_path,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    run_chat_mode(chain)


if __name__ == "__main__":
    main()
