import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "oop_labs"
TOP_K = 3  # how many chunks to retrieve
DISTANCE_THRESHOLD = 1.05


def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Collection loaded: {collection.count()} chunks available.\n")

    while True:
        question = input("Ask a question (or 'quit): ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        # embed the question with the SAME model used for indexing
        query_embedding = model.encode([question]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=TOP_K,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # DEBUG: always show raw results with distances, before filtering
        print("\n[DEBUG] Raw top-K results (before threshold filter):")
        for doc, meta, dist in zip(docs, metas, distances):
            print(
                f"  distance={dist:.4f} | source={meta['source']} | preview={doc[:80]!r}")

        # filter out anything above the threshold
        filtered = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs, metas, distances)
            if dist <= DISTANCE_THRESHOLD
        ]

        if not filtered:
            print("\nNo confident matches found for this question.\n")
            continue

        print(f"\n{len(filtered)} confident result(s):\n" + "-" * 50)
        for i, (doc, meta, dist) in enumerate(filtered):
            print(f"[{i+1}] source: {meta['source']} | distance: {dist:.4f}")
            print(doc[:400])
            print("-" * 50)
        print()


if __name__ == "__main__":
    main()
