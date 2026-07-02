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
        try:
            with fitz.open(doc_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
            return text
        except FileNotFoundError:
            print(f"PDF file not found: {doc_path}")
            return ""
        except Exception as e:
            print(f"Error reading PDF file: {doc_path}")
            print(e)
            return ""

    @staticmethod
    @track_stats
    def process_document(doc_path: str) -> list[Chunk]:
        if doc_path.lower().endswith(".pdf"):
            doc_text = DocumentService.read_pdf_document(doc_path)
        else:
            doc_text = DocumentService.read_document(doc_path)

        splitter = RagService.get_splitter()
        text_chunks = splitter.split_text(doc_text)

        chunks = []
        for chunk in text_chunks:
            emb = RagService.generate_embeddings(chunk)
            chunks.append(Chunk(doc_path, chunk, emb))

        return chunks
