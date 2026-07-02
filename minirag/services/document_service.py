import fitz

from minirag.models import Chunk
from minirag.services.rag_service import RagService
from minirag.utils.stats_utils import track_stats


class DocumentService:
    @staticmethod
    def read_document(doc_path: str) -> str:
        try:
            with open(doc_path, "r") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"File not found: {doc_path}")
            return ""
        except Exception as e:
            print(f"Error reading file: {doc_path}")
            print(e)
            return ""

        return text

    @staticmethod
    def read_pdf_document(doc_path: str) -> str:
        pages = DocumentService.read_pdf_pages(doc_path)
        if not pages:
            return ""

        return "\n".join([page_text for _, page_text in pages]) + "\n"

    @staticmethod
    def read_pdf_pages(doc_path: str) -> list[tuple[int | None, str]]:
        try:
            with fitz.open(doc_path) as doc:
                pages = []
                for page_index, page in enumerate(doc, start=1):
                    pages.append((page_index, page.get_text().strip()))
            return pages
        except FileNotFoundError:
            print(f"PDF file not found: {doc_path}")
            return []
        except Exception as e:
            print(f"Error reading PDF file: {doc_path}")
            print(e)
            return []

    @staticmethod
    def read_document_pages(doc_path: str) -> list[tuple[int | None, str]]:
        if doc_path.lower().endswith(".pdf"):
            return DocumentService.read_pdf_pages(doc_path)

        doc_text = DocumentService.read_document(doc_path)
        if not doc_text:
            return []

        return [(None, doc_text)]

    @staticmethod
    def build_chunk_id(
        doc_path: str,
        chunk_index: int,
        page_number: int | None = None,
    ) -> str:
        if page_number is None:
            return f"{doc_path}#chunk-{chunk_index}"

        return f"{doc_path}#page-{page_number}-chunk-{chunk_index}"

    @staticmethod
    @track_stats
    def process_document(doc_path: str) -> list[Chunk]:
        splitter = RagService.get_splitter()
        chunks = []
        chunk_index = 0

        for page_number, page_text in DocumentService.read_document_pages(doc_path):
            text_chunks = splitter.split_text(page_text)

            for chunk_text in text_chunks:
                emb = RagService.generate_embeddings(chunk_text)
                chunks.append(
                    Chunk(
                        document_name=doc_path,
                        text=chunk_text,
                        embedding=emb,
                        chunk_id=DocumentService.build_chunk_id(
                            doc_path,
                            chunk_index,
                            page_number,
                        ),
                        chunk_index=chunk_index,
                        page_number=page_number,
                    )
                )
                chunk_index += 1

        return chunks
