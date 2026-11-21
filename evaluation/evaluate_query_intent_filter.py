"""Compare baseline hybrid retrieval, cross-encoder reranking,
and cross-encoder reranking with a lightweight intent filter.

This is an isolated evaluation experiment. It does not modify
the production ChromaDB collection or production services.
"""

import re
import uuid
from typing import cast

import chromadb
from chromadb.api.types import Embeddings
from sentence_transformers import CrossEncoder

from evaluation.multi_document_dataset import (
    EVALUATION_DOCUMENTS,
    MULTI_DOCUMENT_EVALUATION_DATASET,
)
from app.services.embedding_service import embedding_service
from app.services.keyword_search_service import keyword_search_service
from app.services.hybrid_search_service import hybrid_search_service
from app.vectorstore.chroma_service import chroma_service

TOP_K = 3
MIN_SIMILARITY = 0.10
CROSS_ENCODER_THRESHOLD = 4.0

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# These terms express requested attributes rather than general topics.
ATTRIBUTE_GROUPS = {
    "age": {"age", "minimum", "old", "young"},
    "cost": {"cost", "costs", "price", "prices", "fee", "fees", "tuition"},
    "requirements": {
        "requirement",
        "requirements",
        "need",
        "needs",
        "required",
        "eligibility",
        "eligible",
    },
    "authority": {
        "authority",
        "department",
        "government",
        "body",
        "responsible",
        "issue",
        "issues",
        "issuing",
        "grant",
        "grants",
    },
    "process": {
        "process",
        "procedure",
        "procedures",
        "complete",
        "completion",
        "automatically",
        "obtain",
        "obtaining",
        "register",
        "registration",
    },
    "comparison": {
        "same",
        "different",
        "difference",
        "compare",
        "comparison",
        "between",
        "versus",
    },
}


def tokenize(text):
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def detect_requested_attributes(question):
    """Detect explicit attributes being requested in a question."""

    question_tokens = tokenize(question)
    detected = set()

    for attribute, terms in ATTRIBUTE_GROUPS.items():
        if question_tokens.intersection(terms):
            detected.add(attribute)

    return detected


def intent_compatible(question, document):
    """Check whether the candidate contains evidence for requested attributes.

    This deliberately uses a conservative lexical check. It is an
    experimental filter, not a semantic classifier.
    """

    requested_attributes = detect_requested_attributes(question)

    if not requested_attributes:
        return True

    document_tokens = tokenize(document)

    # An attribute is supported when at least one of its terms appears
    # in the candidate text.
    supported_attributes = {
        attribute
        for attribute in requested_attributes
        if document_tokens.intersection(ATTRIBUTE_GROUPS[attribute])
    }

    return bool(supported_attributes)


def build_synthetic_corpus():
    ids = []
    documents = []
    metadatas = []

    for source in EVALUATION_DOCUMENTS:
        for index, chunk in enumerate(source["chunks"]):
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])
            metadatas.append(
                {
                    "document_id": source["document_id"],
                    "filename": source["filename"],
                    "chunk_index": index,
                }
            )

    return ids, documents, metadatas


def retrieve_and_score_cases(cross_encoder):
    """Retrieve candidates once and calculate cross-encoder scores once."""

    scored_cases = []

    for index, case in enumerate(
        MULTI_DOCUMENT_EVALUATION_DATASET,
        start=1,
    ):
        question = case["question"]

        candidates = hybrid_search_service.search(
            query=question,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
            return_candidates=True,
        )

        candidate_ids = (candidates.get("ids") or [[]])[0]
        candidate_documents = (candidates.get("documents") or [[]])[0]

        pairs = [(question, document) for document in candidate_documents]

        scores = cross_encoder.predict(pairs) if pairs else []

        scored_candidates = []

        for candidate_index, document in enumerate(candidate_documents):
            scored_candidates.append(
                {
                    "chunk_id": candidate_ids[candidate_index],
                    "document": document,
                    "score": float(scores[candidate_index]),
                }
            )

        scored_candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        scored_cases.append(
            {
                "case": case,
                "candidates": scored_candidates,
            }
        )

        print(f"Scored question {index}/" f"{len(MULTI_DOCUMENT_EVALUATION_DATASET)}")

    return scored_cases


def evaluate_results(case, chunk_ids):
    direct = set(case["relevant_chunk_ids"])
    supporting = set(case["supporting_chunk_ids"])
    useful = direct | supporting

    false_positives = [chunk_id for chunk_id in chunk_ids if chunk_id not in useful]

    if not case["should_find_answer"]:
        return {
            "retrieved": len(chunk_ids),
            "useful": 0,
            "false_positives": len(chunk_ids),
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "negative_correct": len(chunk_ids) == 0,
        }

    useful_count = sum(chunk_id in useful for chunk_id in chunk_ids)

    direct_found = [chunk_id for chunk_id in chunk_ids if chunk_id in direct]

    return {
        "retrieved": len(chunk_ids),
        "useful": useful_count,
        "false_positives": len(false_positives),
        "precision": (useful_count / len(chunk_ids) if chunk_ids else 0.0),
        "recall": (len(set(direct_found)) / len(direct) if direct else 0.0),
        "mrr": next(
            (
                1 / rank
                for rank, chunk_id in enumerate(chunk_ids, start=1)
                if chunk_id in direct
            ),
            0.0,
        ),
        "negative_correct": None,
    }


def run_experiment(scored_cases, mode):
    positive_results = []
    negative_results = []

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {mode.upper()}")
    print("=" * 70)

    for index, entry in enumerate(scored_cases, start=1):
        case = entry["case"]
        candidates = entry["candidates"]

        if mode == "baseline":
            selected = candidates[:TOP_K]

        elif mode == "cross_encoder":
            selected = [
                item for item in candidates if item["score"] >= CROSS_ENCODER_THRESHOLD
            ][:TOP_K]

        elif mode == "intent_filter":
            selected = [
                item
                for item in candidates
                if (
                    item["score"] >= CROSS_ENCODER_THRESHOLD
                    and intent_compatible(
                        case["question"],
                        item["document"],
                    )
                )
            ][:TOP_K]

        else:
            raise ValueError(f"Unknown experiment mode: {mode}")

        if mode == "intent_filter" and case["should_find_answer"]:
            expected_ids = set(case["relevant_chunk_ids"])
            selected_ids = {item["chunk_id"] for item in selected}
            missing = expected_ids - selected_ids

            if missing:
                print(f"WARNING: Relevant chunks filtered out: " f"{sorted(missing)}")

        selected_ids = [item["chunk_id"] for item in selected]

        metrics = evaluate_results(case, selected_ids)

        print(f"\nTest {index}/{len(scored_cases)}")
        print(f"Question: {case['question']}")
        print(f"Selected chunks: {selected_ids}")

        if mode == "intent_filter":
            print(
                "Requested attributes: "
                f"{sorted(detect_requested_attributes(case['question']))}"
            )

            for item in candidates:
                compatible = intent_compatible(
                    case["question"],
                    item["document"],
                )
                print(
                    f"  {item['chunk_id']}: "
                    f"score={item['score']:.4f}, "
                    f"intent_compatible={compatible}"
                )

        if case["should_find_answer"]:
            positive_results.append(metrics)
        else:
            negative_results.append(metrics)
            print(f"Correctly rejected: " f"{metrics['negative_correct']}")

    positive_count = len(positive_results)
    negative_count = len(negative_results)

    total_useful = sum(item["useful"] for item in positive_results)
    total_retrieved = sum(item["retrieved"] for item in positive_results)

    summary = {
        "macro_precision": (
            sum(item["precision"] for item in positive_results) / positive_count
            if positive_count
            else 0.0
        ),
        "micro_precision": (total_useful / total_retrieved if total_retrieved else 0.0),
        "recall": (
            sum(item["recall"] for item in positive_results) / positive_count
            if positive_count
            else 0.0
        ),
        "mrr": (
            sum(item["mrr"] for item in positive_results) / positive_count
            if positive_count
            else 0.0
        ),
        "positive_fp": sum(item["false_positives"] for item in positive_results),
        "negative_fp": sum(item["false_positives"] for item in negative_results),
        "total_fp": (
            sum(item["false_positives"] for item in positive_results)
            + sum(item["false_positives"] for item in negative_results)
        ),
        "negative_correct": sum(item["negative_correct"] for item in negative_results),
        "negative_total": negative_count,
    }

    print("\nEXPERIMENT SUMMARY")
    print(f"Macro precision: {summary['macro_precision']:.2%}")
    print(f"Micro precision: {summary['micro_precision']:.2%}")
    print(f"Mean direct recall: {summary['recall']:.2%}")
    print(f"Mean MRR: {summary['mrr']:.2%}")
    print(f"Positive false positives: {summary['positive_fp']}")
    print(f"Negative false positives: {summary['negative_fp']}")
    print(f"Total false positives: {summary['total_fp']}")
    print(
        f"Negative queries rejected: "
        f"{summary['negative_correct']}/"
        f"{summary['negative_total']}"
    )

    return summary


def evaluate():
    print("\nKLYRA QUERY INTENT FILTER EVALUATION")
    print("===================================")

    print(f"\nLoading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_intent_{uuid.uuid4().hex[:12]}",
        metadata={"hnsw:space": "cosine"},
    )

    original_collection = chroma_service.collection

    try:
        ids, documents, metadatas = build_synthetic_corpus()

        print(f"Synthetic documents: {len(EVALUATION_DOCUMENTS)}")
        print(f"Synthetic chunks: {len(documents)}")
        print("Generating embeddings...")

        embeddings = cast(
            Embeddings,
            embedding_service.generate_embeddings(documents),
        )

        temporary_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        chroma_service.collection = temporary_collection
        keyword_search_service.invalidate_cache()

        print("Temporary corpus indexed successfully.")
        print("\nRetrieving and scoring candidates once...")

        scored_cases = retrieve_and_score_cases(cross_encoder)

        results = {}

        for mode in (
            "baseline",
            "cross_encoder",
            "intent_filter",
        ):
            results[mode] = run_experiment(
                scored_cases,
                mode,
            )

        print("\n\nFINAL EXPERIMENT COMPARISON")
        print("=" * 105)

        print(
            f"{'Method':<20}"
            f"{'Macro P':>11}"
            f"{'Micro P':>11}"
            f"{'Recall':>11}"
            f"{'MRR':>11}"
            f"{'Total FP':>11}"
            f"{'Rejected':>13}"
        )

        for mode, metrics in results.items():
            print(
                f"{mode:<20}"
                f"{metrics['macro_precision']:>10.2%}"
                f"{metrics['micro_precision']:>11.2%}"
                f"{metrics['recall']:>11.2%}"
                f"{metrics['mrr']:>11.2%}"
                f"{metrics['total_fp']:>11}"
                f"{metrics['negative_correct']}/"
                f"{metrics['negative_total']:>10}"
            )

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
