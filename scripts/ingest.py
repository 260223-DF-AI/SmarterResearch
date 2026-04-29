"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
"""

import argparse
import os

from acquire_data import _get_downloaded_papers
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pypdf import PdfReader

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse ingestion CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing PDF/text documents.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="research",
        help="Pinecone namespace to upsert into.",
    )
    return parser.parse_args()


def _read_pdf(file_path: str):
    """
    Handles reading a PDF file.

    Yields page number alongside raw text for each given page.
    """
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages, start=1):
        # layout extraction mode attempts to handle formatted text better
        page_text = page.extract_text(extraction_mode="layout")

        # remove small lines that probably aren't body text
        filtered_lines = [
            line for line in page_text.split("\n") if len(line.strip()) > 30
        ]
        yield (page_num, "\n".join(filtered_lines))


def load_documents(input_dir: str) -> list:
    """
    Load and return raw documents from the input directory.

    - Support PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
    - Support plain text files.
    - Return a list of Document objects with content and metadata
      (source filename, page number).
    """

    docs: list[Document] = []
    for file_name, ext in _get_downloaded_papers(input_dir):
        file_path: str = f"{input_dir}/{file_name}.{ext}"

        raw_text: str = ""

        # read files differently depending on file type
        match ext:
            case "txt":
                with open(file_path, "r") as f:
                    raw_text = f.read()
                    doc = Document(
                        page_content=raw_text, metadata={"source": file_path, "page": 0}
                    )
                    docs.append(doc)

            case "pdf":
                for page_num, content in _read_pdf(file_path):
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "page": page_num},
                    )
                    docs.append(doc)

    return docs


def _add_document_metadata(doc, new_metadata):
    old_metadata = doc.metadata
    new_metadata = {**old_metadata, **new_metadata}
    return Document(page_content=doc.page_content, metadata=new_metadata)


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.
    """

    # RecursiveCharacterTextSplitter

    # following splitter separators sourced from https://medium.com/data-and-beyond/text-splitters-in-langchain-for-data-processing-3a958eea2797
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ".",
        ],
    )

    chunks: list[dict] = []
    chunk_id = 0
    # split documents into chunks
    for document in documents:
        split_text = splitter.split_documents([document])
        for num, chunk in enumerate(split_text):
            chunks.append(
                {
                    "_id": str(chunk_id),
                    "chunk_text": chunk.page_content,
                    **_add_document_metadata(chunk, {"chunk_num": num}).metadata,
                }
            )
            chunk_id += 1

    return chunks


def generate_embeddings_and_upsert(chunks: list, namespace: str) -> list:
    """
    Upserts chunks into Pinecone in the maximum supported batch size (96).

    Pinecone index is configured to automatically generate embeddings.
    """

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    index = pc.Index(name=PINECONE_INDEX_NAME)

    # pine cone can only upload in batches of 96 at most
    batch_size = 96
    for i in range(0, len(chunks), batch_size):
        index.upsert_records(namespace=namespace, records=chunks[i : i + batch_size])


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents)
    generate_embeddings_and_upsert(chunks, args.namespace)

    print(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
