"""Synthetic evaluation corpus for Klyra retrieval testing.

These fixtures are independent of the production ChromaDB collection.
They can be used to test retrieval, deduplication and reranking safely.
"""

EVALUATION_DOCUMENTS = [
    {
        "document_id": "driving_licence",
        "filename": "driving_licence.pdf",
        "chunks": [
            {
                "chunk_id": "driving_licence_0",
                "content": (
                    "To obtain a driving licence in Sri Lanka, an applicant "
                    "must first obtain a learner's permit. Applicants must "
                    "meet the minimum age requirement and provide valid "
                    "identification."
                ),
            },
            {
                "chunk_id": "driving_licence_1",
                "content": (
                    "Applicants for a Sri Lankan driving licence must pass "
                    "a written examination and a practical driving test. "
                    "The Department of Motor Traffic is responsible for "
                    "issuing driving licences."
                ),
            },
        ],
    },
    {
        "document_id": "university_admissions",
        "filename": "university_admissions.pdf",
        "chunks": [
            {
                "chunk_id": "university_admissions_0",
                "content": (
                    "In Sri Lanka, the Advanced Level examination is "
                    "generally used to determine eligibility for university "
                    "admission. Students are assessed based on their "
                    "examination results."
                ),
            },
            {
                "chunk_id": "university_admissions_1",
                "content": (
                    "Sri Lanka has public and private universities. "
                    "Admission requirements vary depending on the "
                    "institution and the degree programme."
                ),
            },
        ],
    },
    {
        "document_id": "vehicle_registration",
        "filename": "vehicle_registration.pdf",
        "chunks": [
            {
                "chunk_id": "vehicle_registration_0",
                "content": (
                    "Vehicle registration in Sri Lanka is a separate "
                    "administrative process from obtaining a driving "
                    "licence. Vehicle owners must follow the applicable "
                    "registration procedures."
                ),
            },
        ],
    },
    {
        "document_id": "information_technology",
        "filename": "information_technology.pdf",
        "chunks": [
            {
                "chunk_id": "information_technology_0",
                "content": (
                    "Information technology includes software development, "
                    "computer networks, databases and information systems. "
                    "Software engineering applies engineering principles "
                    "to the design and maintenance of software."
                ),
            },
        ],
    },
]


MULTI_DOCUMENT_EVALUATION_DATASET = [
    {
        "question": (
            "What are the requirements for obtaining a driving " "licence in Sri Lanka?"
        ),
        "expected_document_ids": ["driving_licence"],
        "expected_terms": [
            "learner's permit",
            "minimum age",
            "valid identification",
            "written examination",
            "practical driving test",
        ],
        "requires_multiple_chunks": True,
        "should_find_answer": True,
        "relevant_chunk_ids": [
            "driving_licence_0",
            "driving_licence_1",
        ],
        "supporting_chunk_ids": [],
    },
    {
        "question": ("Which authority issues driving licences in Sri Lanka?"),
        "expected_document_ids": ["driving_licence"],
        "expected_terms": ["Department of Motor Traffic"],
        "requires_multiple_chunks": False,
        "should_find_answer": True,
        "relevant_chunk_ids": ["driving_licence_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": (
            "How does the Advanced Level examination relate to "
            "university admission in Sri Lanka?"
        ),
        "expected_document_ids": ["university_admissions"],
        "expected_terms": [
            "Advanced Level examination",
            "university admission",
        ],
        "requires_multiple_chunks": False,
        "should_find_answer": True,
        "relevant_chunk_ids": ["university_admissions_0"],
        "supporting_chunk_ids": [],
    },
    {
        "question": ("Does Sri Lanka have both public and private universities?"),
        "expected_document_ids": ["university_admissions"],
        "expected_terms": ["public and private universities"],
        "requires_multiple_chunks": False,
        "should_find_answer": True,
        "relevant_chunk_ids": ["university_admissions_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": (
            "Are vehicle registration and obtaining a driving licence "
            "the same process?"
        ),
        "expected_document_ids": [
            "vehicle_registration",
            "driving_licence",
        ],
        "expected_terms": [
            "separate administrative process",
            "driving licence",
        ],
        "requires_multiple_chunks": True,
        "should_find_answer": True,
        "relevant_chunk_ids": ["vehicle_registration_0"],
        "supporting_chunk_ids": [
            "driving_licence_0",
            "driving_licence_1",
        ],
    },
    {
        "question": ("Who won the FIFA World Cup in 2022?"),
        "expected_document_ids": [],
        "expected_terms": [],
        "requires_multiple_chunks": False,
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": ("Explain quantum physics and quantum computing."),
        "expected_document_ids": [],
        "expected_terms": [],
        "requires_multiple_chunks": False,
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
]
