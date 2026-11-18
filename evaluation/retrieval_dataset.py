
EVALUATION_DATASET = [
    {
        "question": "What are the requirements to obtain a driving licence in Sri Lanka?",
        "expected_terms": [
            "learner's permit",
            "minimum age",
            "valid identification",
            "written examination",
            "practical driving test",
        ],
        "should_find_answer": True,
    },
    {
        "question": "Which department issues driving licences in Sri Lanka?",
        "expected_terms": [
            "Department of Motor Traffic",
        ],
        "should_find_answer": True,
    },
    {
        "question": "How is the Advanced Level examination used in Sri Lanka?",
        "expected_terms": [
            "university admission",
            "public and private universities",
        ],
        "should_find_answer": True,
    },
    {
        "question": "Does Sri Lanka have public and private universities?",
        "expected_terms": [
            "public and private universities",
        ],
        "should_find_answer": True,
    },
    {
        "question": "Who won the FIFA World Cup in 2022?",
        "expected_terms": [],
        "should_find_answer": False,
    },
    {
        "question": "Explain quantum physics and quantum computing.",
        "expected_terms": [],
        "should_find_answer": False,
    },
]
