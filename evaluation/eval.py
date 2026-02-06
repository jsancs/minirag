import numpy as np
from typing import List
import random
import pandas as pd

from minirag.services.rag_service import RagService
from minirag.models import Chunk
from minirag.utils.stats_utils import function_stats


def generate_random_text(length: int = 100) -> str:
    """Generate random text of specified length."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    return " ".join(random.choices(words, k=length))


def generate_synthetic_embedding(dim: int = 384) -> list[float]:
    """Generate synthetic embedding vector of specified dimension."""
    # Generate random unit vector for consistent similarity calculations
    embedding = np.random.normal(0, 1, dim)
    normalized_embedding = embedding / np.linalg.norm(embedding)
    return normalized_embedding.tolist()


def create_synthetic_collection(size: int) -> list[Chunk]:
    """Create a synthetic collection of specified size."""
    collection = []
    for i in range(size):
        text = generate_random_text()
        embedding = generate_synthetic_embedding()
        chunk = Chunk(
            document_name=f"synthetic_doc_{i}", text=text, embedding=embedding
        )
        collection.append(chunk)
    return collection


def evaluate_similarity_search(
    collection_sizes: list[int], num_queries: int = 3, runs: int = 3
):
    """Evaluate similarity search performance for different collection sizes."""
    results = []

    for size in collection_sizes:
        print(f"\nEvaluating collection size: {size}")

        # Create synthetic collection
        collection = create_synthetic_collection(size)

        # Generate synthetic queries (just random embeddings)
        queries = [generate_random_text() for _ in range(num_queries)]

        # Run queries multiple times
        for query_idx, query in enumerate(queries):
            for run in range(runs):
                print(
                    f"Running query {query_idx + 1}/{num_queries}, "
                    f"run {run + 1}/{runs}"
                )

                # Perform similarity search
                RagService.similarity_search(query, collection)

                # Get the latest run statistics
                stats = function_stats["similarity_search"][-1]

                results.append(
                    {
                        "collection_size": size,
                        "query_number": query_idx + 1,
                        "run_number": run + 1,
                        "execution_time": stats["execution_time"],
                        "memory_used": stats["memory_used"],
                    }
                )

    return pd.DataFrame(results)


def main():
    # Define collection sizes to test (you can adjust these)
    collection_sizes = [100, 500, 1000, 5000, 10_000, 20_000, 50_000]

    # Run evaluation
    results_df = evaluate_similarity_search(collection_sizes)

    # Calculate and display aggregate statistics
    summary = results_df.groupby("collection_size").agg(
        {"execution_time": ["mean", "std"], "memory_used": ["mean", "std"]}
    )

    print("\nPerformance Summary:")
    print(summary)

    # Save results to CSV
    results_df.to_csv(
        "evaluation/results/similarity_search_performance.csv", index=False
    )
    print(
        "\nDetailed results saved to 'evaluation/results/similarity_search_performance.csv'"
    )


if __name__ == "__main__":
    main()
