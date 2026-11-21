"""Evaluate semantic entailment filtering after cross-encoder retrieval.

Compares:
1. Cross-encoder reranking with threshold 4.0.
2. Cross-encoder reranking plus NLI entailment verification.

Uses an isolated temporary ChromaDB collection.
Production services, models and stored documents are not modified.
"""

import re
import uuid

import chromadb
import numpy as np
import torch
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
ENTAILMENT_THRESHOLD = 0.70

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


# Human-annotated answer claims for evaluation.
# Each candidate passage is tested as the premise against these claims.
# Multiple claims allow separate chunks to support different parts
# of a multi-part question.
#
# These are evaluation annotations, not production query rules.

ANSWER_CLAIMS = {
    1: [
        "An applicant must obtain a learner's permit before obtaining a driving licence in Sri Lanka.",
        "Driving licence applicants in Sri Lanka must pass a written examination and a practical driving test.",
    ],
    2: [
        "The Department of Motor Traffic is responsible for issuing driving licences.",
    ],
    3: [
        "Advanced Level examination results are used to determine eligibility for university admission in Sri Lanka.",
    ],
    4: [
        "Sri Lanka has both public universities and private universities.",
    ],
    5: [
        "Vehicle registration in Sri Lanka is a separate administrative process from obtaining a driving licence.",
    ],
    6: [
        "The supplied passages identify the winner of the 2022 FIFA World Cup.",
    ],
    7: [
        "The supplied passages explain quantum physics and quantum computing.",
    ],
    8: [
        "An applicant must obtain a learner's permit before obtaining a driving licence in Sri Lanka.",
        "Driving licence applicants in Sri Lanka must pass a written examination and a practical driving test.",
    ],
    9: [
        "The Department of Motor Traffic is responsible for issuing driving licences.",
    ],
    10: [
        "Advanced Level examination results affect eligibility for university admission in Sri Lanka.",
    ],
    11: [
        "Information technology includes software development, computer networks, databases and information systems.",
        "Software engineering applies engineering principles to the design and maintenance of software.",
    ],
    12: [
        "Passing a driving test does not automatically complete vehicle registration in Sri Lanka.",
    ],
    13: [
        "The supplied passages specify the minimum age required to register a vehicle in Sri Lanka.",
    ],
    14: [
        "The supplied passages specify the cost of a software engineering degree in Sri Lanka.",
    ],
    15: [
        "The supplied passages explain how to renew a passport.",
    ],
}


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


def retrieve_candidates(cross_encoder):
    """Retrieve and cross-encoder-score candidates once per question."""

    scored_cases = []

    for index, case in enumerate(
        MULTI_DOCUMENT_EVALUATION_DATASET,
        start=1,
    ):
        question = case["question"]

        results = hybrid_search_service.search(
            query=question,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
            return_candidates=True,
        )

        candidate_ids = (results.get("ids") or [[]])[0]
        candidate_documents = (results.get("documents") or [[]])[0]

        pairs = [(question, document) for document in candidate_documents]

        scores = cross_encoder.predict(pairs) if pairs else []

        candidates = []

        for candidate_index, document in enumerate(candidate_documents):
            candidates.append(
                {
                    "chunk_id": candidate_ids[candidate_index],
                    "document": document,
                    "cross_score": float(scores[candidate_index]),
                }
            )

        candidates.sort(
            key=lambda item: item["cross_score"],
            reverse=True,
        )

        scored_cases.append(
            {
                "case": case,
                "candidates": candidates,
            }
        )

        print(
            f"Retrieved question {index}/" f"{len(MULTI_DOCUMENT_EVALUATION_DATASET)}"
        )

    return scored_cases


def get_entailment_index(nli_model):
    """Find the entailment output index from the model configuration."""

    labels = nli_model.model.config.id2label

    for index, label in labels.items():
        if "entail" in str(label).lower():
            return int(index)

    raise ValueError(f"Could not identify the entailment label. Model labels: {labels}")


def calculate_entailment_probability(nli_model, entailment_index, pairs):
    """Return softmax-normalized entailment probabilities."""

    if not pairs:
        return []

    logits = np.asarray(
        nli_model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    if logits.shape[1] <= entailment_index:
        raise ValueError(f"Unexpected NLI output shape: {logits.shape}")

    # Numerically stable softmax.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )

    return probabilities[:, entailment_index].tolist()


def calculate_sentence_level_entailment(
    nli_model,
    passage,
    hypotheses,
):
    """
    Evaluate each sentence independently against the hypotheses.
    Return the highest entailment probability.
    """

    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())

    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences or not hypotheses:
        return 0.0

    pairs = [
        (sentence, hypothesis) for sentence in sentences for hypothesis in hypotheses
    ]

    logits = nli_model.predict(
        pairs,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()

    entailment_index = get_entailment_index(nli_model)

    return float(probabilities[:, entailment_index].max())


def score_entailment(nli_model, entailment_index, scored_cases):
    """Score candidates using sentence-level NLI verification."""

    print("\nScoring sentence-level semantic entailment...")

    total_candidates = 0

    for case_index, entry in enumerate(scored_cases, start=1):
        claims = ANSWER_CLAIMS[case_index]
        candidates = entry["candidates"]

        for candidate in candidates:
            candidate["entailment_score"] = calculate_sentence_level_entailment(
                nli_model,
                candidate["document"],
                claims,
            )

        total_candidates += len(candidates)

        print(f"Scored question " f"{case_index}/{len(scored_cases)}")

    print(f"Total candidate passages checked: {total_candidates}")


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

        cross_encoder_candidates = [
            candidate
            for candidate in candidates
            if candidate["cross_score"] >= CROSS_ENCODER_THRESHOLD
        ]

        if mode == "cross_encoder":
            selected = cross_encoder_candidates[:TOP_K]

        elif mode == "semantic_entailment":
            selected = [
                candidate
                for candidate in cross_encoder_candidates
                if candidate["entailment_score"] >= ENTAILMENT_THRESHOLD
            ][:TOP_K]

        else:
            raise ValueError(f"Unknown experiment mode: {mode}")

        selected_ids = [candidate["chunk_id"] for candidate in selected]

        metrics = evaluate_results(case, selected_ids)

        print(f"\nTest {index}/{len(scored_cases)}")
        print(f"Question: {case['question']}")
        print(f"Selected chunks: {selected_ids}")

        if mode == "semantic_entailment":
            print("Candidate verification:")

            for candidate in candidates:
                selected_status = "SELECTED" if candidate in selected else "FILTERED"

                print(
                    f"  {candidate['chunk_id']}: "
                    f"cross={candidate['cross_score']:.4f}, "
                    f"entailment={candidate['entailment_score']:.4f}, "
                    f"[{selected_status}]"
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


def diagnose_nli_predictions(nli_model):
    """Inspect raw NLI outputs for two important evaluation cases."""

    diagnostics = [
        (
            "Test 5 - First sentence only",
            (
                "Vehicle registration in Sri Lanka is a separate "
                "administrative process from obtaining a driving licence."
            ),
            ANSWER_CLAIMS[5][0],
        ),
        (
            "Test 5 - Second sentence only",
            ("Vehicle owners must follow the applicable " "registration procedures."),
            ANSWER_CLAIMS[5][0],
        ),
        (
            "Test 13 - First sentence only",
            (
                "To obtain a driving licence in Sri Lanka, an applicant "
                "must first obtain a learner's permit."
            ),
            ANSWER_CLAIMS[13][0],
        ),
        (
            "Test 13 - Second sentence only",
            (
                "Applicants must meet the minimum age requirement "
                "and provide valid identification."
            ),
            ANSWER_CLAIMS[13][0],
        ),
    ]

    pairs = [(premise, hypothesis) for _, premise, hypothesis in diagnostics]

    logits = nli_model.predict(
        pairs,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    print("\nSENTENCE-LEVEL NLI DIAGNOSTICS")

    for index, (description, premise, hypothesis) in enumerate(diagnostics):
        print(f"\n{description}")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Raw logits: {logits[index]}")


def evaluate():
    print("\nKLYRA SEMANTIC ENTAILMENT EVALUATION")
    print("===================================")

    print(f"\nLoading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    print(f"\nLoading NLI model: {NLI_MODEL}")
    nli_model = CrossEncoder(NLI_MODEL)

    entailment_index = get_entailment_index(nli_model)

    diagnose_nli_predictions(nli_model)

    print(f"Detected entailment output index: " f"{entailment_index}")

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_entailment_{uuid.uuid4().hex[:12]}",
        metadata={"hnsw:space": "cosine"},
    )

    original_collection = chroma_service.collection

    try:
        ids, documents, metadatas = build_synthetic_corpus()

        print(f"\nSynthetic documents: {len(EVALUATION_DOCUMENTS)}")
        print(f"Synthetic chunks: {len(documents)}")
        print("Generating embeddings...")

        embeddings = np.asarray(
            embedding_service.generate_embeddings(documents),
            dtype=np.float32,
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

        scored_cases = retrieve_candidates(cross_encoder)

        score_entailment(
            nli_model,
            entailment_index,
            scored_cases,
        )

        results = {}

        for mode in (
            "cross_encoder",
            "semantic_entailment",
        ):
            results[mode] = run_experiment(
                scored_cases,
                mode,
            )

        print("\n\nFINAL EXPERIMENT COMPARISON")
        print("=" * 105)

        print(
            f"{'Method':<23}"
            f"{'Macro P':>11}"
            f"{'Micro P':>11}"
            f"{'Recall':>11}"
            f"{'MRR':>11}"
            f"{'Total FP':>11}"
            f"{'Rejected':>13}"
        )

        for mode, metrics in results.items():
            print(
                f"{mode:<23}"
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
