"""Test cross-encoder sensitivity to alternative question phrasings.

Scores fixed question/chunk pairs directly.
Does not use or modify ChromaDB or production retrieval.
"""

from sentence_transformers import CrossEncoder

from evaluation.cross_encoder_robustness_dataset import EVALUATION_DOCUMENTS


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


QUESTION_GROUPS = {
    "Software development": {
        "chunks": [
            "software_development_0",
            "software_development_1",
        ],
        "questions": [
            "What is the process for building and maintaining software?",
            "What stages make up the software development lifecycle?",
            "What steps are involved in developing software?",
            "How is software designed, developed, tested and maintained?",
            "What activities take place throughout the software development process?",
            "How do developers create and maintain software applications?",
        ],
    },
    "Internet networking": {
        "chunks": [
            "networking_0",
            "networking_1",
        ],
        "questions": [
            "How does the internet move information between networks?",
            "How do routers forward traffic between different networks?",
            "How are data packets transferred across interconnected networks?",
            "How do computer networks communicate with one another?",
            "How does information travel from one network to another?",
            "What role do IP and routing play in transferring network data?",
        ],
    },
}


def load_chunks():
    chunks = {}

    for document in EVALUATION_DOCUMENTS:
        for chunk in document["chunks"]:
            chunks[chunk["chunk_id"]] = chunk["content"]

    return chunks


def evaluate():
    chunks = load_chunks()

    print(f"Loading model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME)

    for group_name, group in QUESTION_GROUPS.items():
        print("\n" + "=" * 100)
        print(group_name.upper())
        print("=" * 100)

        pairs = []

        for question in group["questions"]:
            for chunk_id in group["chunks"]:
                pairs.append((question, chunks[chunk_id]))

        scores = model.predict(pairs)

        score_index = 0

        for question in group["questions"]:
            print(f"\nQuestion: {question}")

            for chunk_id in group["chunks"]:
                score = float(scores[score_index])
                score_index += 1

                print(f"  {chunk_id:<25} Score: {score:>9.4f}")

    print("\nExperiment completed.")
    print("No ChromaDB collections or production data were modified.")


if __name__ == "__main__":
    evaluate()
