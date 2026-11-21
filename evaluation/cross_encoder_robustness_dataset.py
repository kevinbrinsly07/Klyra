"""Expanded synthetic dataset for cross-encoder robustness evaluation."""

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
                    "Applicants must pass a written examination and a "
                    "practical driving test. The Department of Motor Traffic "
                    "is responsible for issuing driving licences."
                ),
            },
            {
                "chunk_id": "driving_licence_2",
                "content": (
                    "After passing the required examinations, applicants "
                    "must complete the remaining licensing procedures. "
                    "The driving licence confirms that the holder is "
                    "authorized to operate the relevant vehicle category."
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
                    "admission. Students are assessed based on examination "
                    "results."
                ),
            },
            {
                "chunk_id": "university_admissions_1",
                "content": (
                    "Sri Lanka has public and private universities. "
                    "Admission requirements vary depending on the "
                    "institution and degree programme."
                ),
            },
            {
                "chunk_id": "university_admissions_2",
                "content": (
                    "University applicants should review the admission "
                    "criteria for their selected degree programme. "
                    "Selection may depend on examination performance "
                    "and programme-specific requirements."
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
            {
                "chunk_id": "vehicle_registration_1",
                "content": (
                    "Vehicle registration records identify a vehicle "
                    "and its registration details. Registration and "
                    "authorization to drive are separate matters."
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
            {
                "chunk_id": "information_technology_1",
                "content": (
                    "Software developers use programming languages, "
                    "development frameworks, version control systems "
                    "and testing tools to build and maintain applications."
                ),
            },
            {
                "chunk_id": "information_technology_2",
                "content": (
                    "Database systems store and organize application data. "
                    "Computer networks allow connected devices and systems "
                    "to communicate and exchange information."
                ),
            },
        ],
    },
    {
        "document_id": "software_development",
        "filename": "software_development.pdf",
        "chunks": [
            {
                "chunk_id": "software_development_0",
                "content": (
                    "The software development lifecycle commonly includes "
                    "requirements analysis, planning, design, implementation, "
                    "testing, deployment and maintenance."
                ),
            },
            {
                "chunk_id": "software_development_1",
                "content": (
                    "Agile development uses iterative work cycles and "
                    "continuous feedback. Scrum is an Agile framework "
                    "that organizes work into time-limited sprints."
                ),
            },
        ],
    },
    {
        "document_id": "database_management",
        "filename": "database_management.pdf",
        "chunks": [
            {
                "chunk_id": "database_management_0",
                "content": (
                    "Relational databases organize information into tables "
                    "containing rows and columns. SQL is used to query, "
                    "insert, update and delete relational data."
                ),
            },
            {
                "chunk_id": "database_management_1",
                "content": (
                    "Database normalization reduces unnecessary data "
                    "duplication and helps maintain data consistency. "
                    "Primary keys identify records and foreign keys "
                    "represent relationships between tables."
                ),
            },
        ],
    },
    {
        "document_id": "cybersecurity",
        "filename": "cybersecurity.pdf",
        "chunks": [
            {
                "chunk_id": "cybersecurity_0",
                "content": (
                    "Cybersecurity protects computer systems, networks "
                    "and data against unauthorized access, attacks "
                    "and damage."
                ),
            },
            {
                "chunk_id": "cybersecurity_1",
                "content": (
                    "Multi-factor authentication adds an additional "
                    "verification step when users sign in. Encryption "
                    "helps protect information from unauthorized reading."
                ),
            },
        ],
    },
    {
        "document_id": "cloud_computing",
        "filename": "cloud_computing.pdf",
        "chunks": [
            {
                "chunk_id": "cloud_computing_0",
                "content": (
                    "Cloud computing provides computing resources such "
                    "as servers, storage and databases over a network. "
                    "Organizations can use cloud services without "
                    "maintaining all physical infrastructure themselves."
                ),
            },
            {
                "chunk_id": "cloud_computing_1",
                "content": (
                    "Infrastructure as a Service provides virtualized "
                    "computing infrastructure. Platform as a Service "
                    "provides managed environments for application "
                    "development and deployment."
                ),
            },
        ],
    },
    {
        "document_id": "project_management",
        "filename": "project_management.pdf",
        "chunks": [
            {
                "chunk_id": "project_management_0",
                "content": (
                    "Project management involves planning, organizing "
                    "and monitoring resources to achieve defined project "
                    "objectives within agreed constraints."
                ),
            },
            {
                "chunk_id": "project_management_1",
                "content": (
                    "Project risks are uncertain events that may affect "
                    "scope, schedule, cost or quality. Risk management "
                    "includes identifying, assessing and responding "
                    "to potential risks."
                ),
            },
        ],
    },
    {
        "document_id": "networking",
        "filename": "networking.pdf",
        "chunks": [
            {
                "chunk_id": "networking_0",
                "content": (
                    "Computer networks connect devices so they can "
                    "communicate and share resources. Routers forward "
                    "network traffic between different networks."
                ),
            },
            {
                "chunk_id": "networking_1",
                "content": (
                    "The Transmission Control Protocol provides reliable "
                    "data delivery between applications. The Internet "
                    "Protocol handles addressing and routing packets."
                ),
            },
        ],
    },
]


MULTI_DOCUMENT_EVALUATION_DATASET = [
    # Answerable questions: direct retrieval
    {
        "question": "What must someone do to obtain a driving licence in Sri Lanka?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["driving_licence_0", "driving_licence_1"],
        "supporting_chunk_ids": ["driving_licence_2"],
    },
    {
        "question": "Which department issues driving licences?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["driving_licence_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What examination is used for university admission in Sri Lanka?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["university_admissions_0"],
        "supporting_chunk_ids": ["university_admissions_2"],
    },
    {
        "question": "Are there public and private universities in Sri Lanka?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["university_admissions_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the difference between vehicle registration and a driving licence?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["vehicle_registration_0"],
        "supporting_chunk_ids": ["vehicle_registration_1", "driving_licence_0"],
    },
    {
        "question": "What areas are included in information technology?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["information_technology_0"],
        "supporting_chunk_ids": ["information_technology_2"],
    },
    {
        "question": "What stages are included in the software development lifecycle?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["software_development_0"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is Scrum and how does it relate to Agile?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["software_development_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How is information organized in a relational database?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["database_management_0"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the purpose of database normalization?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["database_management_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How does multi-factor authentication improve security?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["cybersecurity_1"],
        "supporting_chunk_ids": ["cybersecurity_0"],
    },
    {
        "question": "What is cloud computing?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["cloud_computing_0"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What does Infrastructure as a Service provide?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["cloud_computing_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What activities are involved in project management?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["project_management_0"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How should project risks be managed?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["project_management_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the role of a router in a computer network?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["networking_0"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "Which protocol provides reliable data delivery?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["networking_1"],
        "supporting_chunk_ids": [],
    },
    # Paraphrased answerable questions
    {
        "question": "What do I need before I can legally drive in Sri Lanka?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["driving_licence_0", "driving_licence_1"],
        "supporting_chunk_ids": ["driving_licence_2"],
    },
    {
        "question": "Which government body is responsible for granting driving licences?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["driving_licence_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How do A-Level results affect entry into universities?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["university_admissions_0"],
        "supporting_chunk_ids": ["university_admissions_2"],
    },
    {
        "question": "What is the process for building and maintaining software?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["software_development_0"],
        "supporting_chunk_ids": ["information_technology_1"],
    },
    {
        "question": "How can a company protect its data from unauthorized access?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["cybersecurity_0", "cybersecurity_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the difference between SQL and database keys?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["database_management_0", "database_management_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How do cloud platforms help companies run applications?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["cloud_computing_0", "cloud_computing_1"],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How does the internet move information between networks?",
        "should_find_answer": True,
        "relevant_chunk_ids": ["networking_0", "networking_1"],
        "supporting_chunk_ids": [],
    },
    # Unsupported questions with lexical overlap
    {
        "question": "What is the minimum age required to register a vehicle?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How much does a software engineering degree cost in Sri Lanka?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the registration fee for a driving licence?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "Which university has the highest admission rate?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the salary of a software developer in Sri Lanka?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "Which database is the fastest for a banking application?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "Which cybersecurity company is the best in Sri Lanka?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the monthly price of cloud computing services?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How many days does a software project usually take?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the maximum speed allowed on Sri Lankan roads?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    # Completely unrelated unsupported questions
    {
        "question": "Who won the FIFA World Cup in 2022?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "Explain quantum physics and quantum computing.",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How do I renew my passport?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "What is the current weather in Colombo?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "Who is the president of the United States?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
    {
        "question": "How can I cook traditional Sri Lankan rice and curry?",
        "should_find_answer": False,
        "relevant_chunk_ids": [],
        "supporting_chunk_ids": [],
    },
]


# Compatibility aliases for scripts using the previous dataset naming.
ROBUSTNESS_EVALUATION_DATASET = MULTI_DOCUMENT_EVALUATION_DATASET
