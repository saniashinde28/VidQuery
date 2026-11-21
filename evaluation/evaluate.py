
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from core.vector_store import load_vector_store
from core.rag_engine import (
    load_rag_chain,
    TOP_K,
    FALLBACK_ANSWER
)


DATASET_PATH = Path(__file__).parent / "test_questions.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_retrieval_metrics(retrieved_docs, relevant_chunk_ids):

    relevant_ids = set(relevant_chunk_ids)

    retrieved_ids = [
        doc.metadata.get("chunk_index")
        for doc in retrieved_docs
    ]

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_ids
    ]

    # Precision@K
    precision = len(relevant_retrieved) / TOP_K

    # Recall@K
    recall = (
        len(set(relevant_retrieved)) / len(relevant_ids)
        if relevant_ids else 0
    )

    # Mean Reciprocal Rank
    reciprocal_rank = 0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "retrieved_ids": retrieved_ids
    }


def evaluate_retrieval(vector_store, dataset):

    precision_scores = []
    recall_scores = []
    mrr_scores = []

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for item in dataset:

        relevant_ids = item["relevant_chunk_ids"]

        # Skip unanswerable questions for retrieval metrics
        if not relevant_ids:
            continue

        docs = vector_store.similarity_search(
            item["question"],
            k=TOP_K
        )

        metrics = calculate_retrieval_metrics(
            docs,
            relevant_ids
        )

        precision_scores.append(metrics["precision"])
        recall_scores.append(metrics["recall"])
        mrr_scores.append(metrics["mrr"])

        print(f"\nQuestion: {item['question']}")
        print(f"Relevant chunks: {relevant_ids}")
        print(f"Retrieved chunks: {metrics['retrieved_ids']}")
        print(f"Precision@{TOP_K}: {metrics['precision']:.3f}")
        print(f"Recall@{TOP_K}: {metrics['recall']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")

    if not precision_scores:
        print("No answerable questions available for retrieval evaluation.")
        return

    print("\n--- Average Retrieval Metrics ---")

    print(
        f"Precision@{TOP_K}: "
        f"{sum(precision_scores) / len(precision_scores):.3f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{sum(recall_scores) / len(recall_scores):.3f}"
    )

    print(
        f"MRR: "
        f"{sum(mrr_scores) / len(mrr_scores):.3f}"
    )


def evaluate_answers(rag_chain, dataset):

    keyword_scores = []
    correct_abstentions = 0
    abstention_count = 0

    print("\n" + "=" * 60)
    print("ANSWER EVALUATION")
    print("=" * 60)

    for item in dataset:

        question = item["question"]

        result = rag_chain.invoke(question)

        answer = result["answer"]

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")

        if item["should_abstain"]:

            abstention_count += 1

            correctly_abstained = (
                answer.strip() == FALLBACK_ANSWER
            )

            if correctly_abstained:
                correct_abstentions += 1

            print(
                "Correct abstention:",
                correctly_abstained
            )

        else:

            keywords = item["expected_keywords"]

            matched = sum(
                keyword.lower() in answer.lower()
                for keyword in keywords
            )

            coverage = (
                matched / len(keywords)
                if keywords else 0
            )

            keyword_scores.append(coverage)

            print(f"Keyword coverage: {coverage:.3f}")

    print("\n--- Average Answer Metrics ---")

    if keyword_scores:
        print(
            "Average keyword coverage:",
            f"{sum(keyword_scores) / len(keyword_scores):.3f}"
        )

    if abstention_count:
        print(
            "Abstention accuracy:",
            f"{correct_abstentions / abstention_count:.3f}"
        )


def main():

    dataset = load_dataset()

    # Load existing Chroma collection
    vector_store = load_vector_store()

    # Evaluate retrieval independently
    evaluate_retrieval(vector_store, dataset)

    # Evaluate generated answers and abstention
    rag_chain = load_rag_chain()

    evaluate_answers(rag_chain, dataset)


if __name__ == "__main__":
    main()
