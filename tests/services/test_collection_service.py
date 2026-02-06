import pytest
from pathlib import Path
import shutil
import tempfile
import os
import numpy as np
from unittest.mock import patch

from minirag.services.collection_service import CollectionService
from minirag.models import Chunk
from minirag.services import rag_service


class TestCollectionService:
    @pytest.fixture
    def temp_dir(self):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after tests
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def collection_service(self, temp_dir):
        return CollectionService(storage_path=temp_dir)

    @pytest.fixture
    def sample_text_files(self, temp_dir):
        # Create sample text files for testing
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()

        file1_path = docs_dir / "test1.txt"
        file2_path = docs_dir / "test2.txt"

        file1_path.write_text("This is test document 1.")
        file2_path.write_text("This is test document 2.")

        return [str(file1_path), str(file2_path), str(docs_dir)]

    def test_init(self, collection_service):
        assert collection_service.active_collection is None
        assert isinstance(collection_service.storage_path, Path)

    def test_create_collection(self, collection_service, sample_text_files):
        # Get the docs directory from sample_text_files
        docs_dir = sample_text_files[-1]
        collection_name = "folder_collection"

        # Mock the embedding generation to avoid external dependencies
        with patch.object(
            rag_service.RagService, "generate_embeddings", return_value=[0.1, 0.2, 0.3]
        ):
            # Create collection from folder
            collection_service.create_collection(
                doc_paths=[docs_dir],  # Pass the directory path
                collection_name=collection_name,
            )

        # Verify collection file exists
        collection_file = (
            Path(collection_service.storage_path) / f"{collection_name}.npy"
        )
        assert collection_file.exists()

        # Load and verify collection contents
        collection_service.load_collection(collection_name)
        assert collection_service.active_collection is not None
        assert isinstance(collection_service.active_collection, list)
        assert all(
            isinstance(chunk, Chunk) for chunk in collection_service.active_collection
        )

        # Verify that chunks from both files in the folder were processed
        doc_names = {
            Path(chunk.document_name).name
            for chunk in collection_service.active_collection
        }
        assert "test1.txt" in doc_names
        assert "test2.txt" in doc_names

    def test_store_embeddings(self, tmp_path):
        # Initialize service with a non-existent directory inside tmp_path
        storage_dir = tmp_path / "new_collections"
        collection_service = CollectionService(storage_path=str(storage_dir))

        # Create a simple test chunk
        test_chunk = Chunk(
            document_name="test_doc.txt",
            text="test content",
            embedding=[1.0, 2.0],
        )

        # Verify directory doesn't exist initially
        assert not storage_dir.exists()

        # Store embeddings
        collection_service._store_embeddings([test_chunk], "test_collection")

        # Verify directory was created
        assert storage_dir.exists()
        assert storage_dir.is_dir()

    def test_load_nonexistent_collection(self, collection_service):
        collection_service.load_collection("nonexistent")
        assert collection_service.active_collection is None

    def test_list_collections(self, collection_service, sample_text_files, capsys):
        # Create a few collections first
        with patch.object(
            rag_service.RagService, "generate_embeddings", return_value=[0.1, 0.2, 0.3]
        ):
            collection_service.create_collection(
                doc_paths=[sample_text_files[0]], collection_name="test1"
            )
            collection_service.create_collection(
                doc_paths=[sample_text_files[1]], collection_name="test2"
            )

        # Call the method we're testing
        collection_service.list_collections()

        # Capture the printed output
        captured = capsys.readouterr()

        # Verify the output contains the expected text
        assert "Available collections:" in captured.out
        assert "test1" in captured.out
        assert "test2" in captured.out
