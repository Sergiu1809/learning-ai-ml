import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "oop_labs"
TOP_K = 3
DISTANCE_THRESHOLD = 1.05
MODEL_NAME = "claude-sonnet-4-5"

client_anthropic = Anthropic()


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant answering questions based ONLY on the provided context from OOP/Java lab documents.

Context:
{context_text}

Question: {question}

Instructions:
- Answer using ONLY the information in the context above.
- If the context does not contain enough information to answer, say so explicitly — do not make up an answer.
- Be concise and accurate."""


def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    print(f"Collection loaded: {collection.count()} chunks available.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        query_embedding = embed_model.encode([question]).tolist()
        results = collection.query(
            query_embeddings=query_embedding, n_results=TOP_K)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        filtered = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs, metas, distances)
            if dist <= DISTANCE_THRESHOLD
        ]

        if not filtered:
            print("\nNo relevant context found in the corpus for this question.\n")
            continue

        context_chunks = [doc for doc, meta, dist in filtered]
        sources = {meta["source"] for doc, meta, dist in filtered}

        prompt = build_prompt(question, context_chunks)

        print("\nGenerating answer...\n")
        response = client_anthropic.messages.create(
            model=MODEL_NAME,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text
        print("Answer:")
        print(answer)
        print(f"\nSources: {', '.join(sources)}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
