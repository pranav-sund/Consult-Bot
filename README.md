
Consultancy & Advisory – Intelligent Chatbot System

A Hybrid Rule-Based, Semantic Search, and AI-Enhanced Advisory Assistant

Overview
The Consultancy and Advisory Chatbot System is a comprehensive conversational platform designed to streamline and automate communication within consultancy firms. It focuses on improving client interactions by enabling real-time FAQ responses, handling service and pricing inquiries, supporting appointment scheduling, generating chat summaries in PDF format, and maintaining complete chat logs. Administrators can manage content through a protected interface that allows editing FAQs and reviewing appointments and chat activity.
The system is built on Python, Flask, and SQLite, adopting a modular architecture that can seamlessly evolve into a highly intelligent advisory platform powered by FastAPI, LangChain, FAISS, and vector embeddings. This README explains both the production-ready Flask system and the advanced AI-driven FastAPI system that represents the next stage of evolution.

1. Current System (Flask + SQLite + Streamlit)
1.1 Core Features
The current version enables clients to register securely and interact with a real-time rule-based chatbot capable of answering predefined FAQs regarding services, pricing, and availability. Clients can also schedule appointments, receive automated email confirmations, and download complete chat interactions as PDF summaries. From the administrative side, authorized personnel can log in through a protected portal, edit or update FAQs, and review both chat logs and appointment records. These features are supported by a rule-based chatbot engine and a well-structured SQLite backend that stores all relevant data, including clients, FAQs, chat logs, and appointment schedules. The system also incorporates secure password hashing, automatic PDF creation, email delivery, and an architecture structured for future scalability.

2. Future System Expansion – AI, Embeddings & Enhanced Logic
As the system evolves, it will shift from purely rule-based logic toward a more sophisticated hybrid advisory model. The first mode in this expanded version will still rely on deterministic rule-based responses sourced from CSV lookup files. The second mode will enhance the system significantly by introducing semantic search capabilities based on embeddings generated through models such as OpenAI or locally hosted alternatives. These embeddings will be processed and indexed through FAISS, enabling more accurate and natural language understanding. The third planned mode merges both rule-based and AI-driven responses into a hybrid advisory assistant that can handle more complex user needs. In this mode, the system not only gives contextual answers but also manages appointments, logs conversations into a SQLite database, generates PDF transcripts, and reloads knowledge bases dynamically from CSV files.

3. Dataset Information for the Future AI System
The system uses a dataset named Chatbot_System_Merged_30Rows.csv, which contains a consolidated structure designed to support both rule-based and semantic search workflows. This file includes fields such as questions, answers, client identifiers, logged queries, system-generated responses, appointment records, time and date information, and administrative authentication values. Because of the structured nature of this dataset, it becomes an essential component for knowledge retrieval and AI-enhanced processing in the future system.

4. System Architecture
4.1 Production Architecture (Flask Version)
The system follows a three-tier design. The presentation layer contains all user interface elements, including HTML, CSS, Bootstrap, and Jinja2 templates, which collectively handle all client and admin interactions visually. The application layer is governed by a Flask backend and manages tasks such as chatbot communication, authentication, FAQ retrieval, appointment processing, email notifications, and PDF generation. The data layer uses SQLite to store clients, FAQs, chat transcripts, appointment details, and administrative credentials. This layered structure ensures a clear separation of responsibilities and maintains reliability and modularity across the entire system.

5. Technology Stack
The current implementation relies on Python 3.8 or later, the Flask framework, SQLite with SQLAlchemy for ORM support, ReportLab for PDF creation, Flask-Mail for sending notifications, Bootstrap 5 for modern interface styling, and secure password hashing utilities such as Werkzeug or bcrypt. The future AI version extends this foundation by incorporating FastAPI for high-performance API handling, FAISS for vector storage and similarity search, LangChain for orchestration of language models, embedding generators from OpenAI or local alternatives, Pandas for data manipulation, and Uvicorn as an asynchronous server.

6. Testing Summary
The system has undergone extensive testing across multiple layers. Unit tests have been performed on chatbot logic, database operations, and appointment scheduling routines. Integration testing verified that chats flow correctly into the database and proceed smoothly into email and PDF generation processes. Security testing included hashed password validation, SQL injection prevention, and CSRF protection. Full system tests were conducted across various browsers and confirmed that all major components behave consistently and successfully under typical usage conditions.

7. Security Measures
Security is implemented throughout the platform. Passwords are encrypted using secure hashing methods, while CSRF protection safeguards user operations against malicious submissions. Role-based access control differentiates privileges between clients and administrators. Cookies are marked as HttpOnly and SameSite to prevent unauthorized manipulation. HTTPS is supported through Flask-Talisman during deployment. SQL injection attempts are blocked through the ORM-based database layer, and input validation is applied at all major data collection points.

8. Installation & Setup
To install the system, begin by cloning the project repository and navigating into the project directory. Create and activate a virtual environment appropriate for your operating system. After that, install all necessary dependencies through the package requirements file. Place the dataset file Chatbot_System_Merged_30Rows.csv into the appropriate directory if you intend to use the future AI features. The Flask system can be started with a simple Python command, after which the application becomes accessible through a browser at the local host address. The AI-enhanced FastAPI version can be launched using Uvicorn, enabling access to advanced endpoints and features.

9. Future FastAPI API Endpoints
The future version of the system will contain dedicated endpoints for managing chatbot interactions, creating appointments, retrieving chat logs, generating PDF summaries, and reloading the knowledge base. These endpoints are designed to offer programmatic access to all core features of the AI-based system and will serve as the foundation for both internal operations and external integrations.

10. Future Enhancements
Several improvements are planned for future releases. The system will eventually incorporate full AI-driven advisory functionality, multilingual capabilities, and a voice-enabled chatbot interface. A modern React-based frontend is expected to enhance the user experience. There will also be an administrative interface for controlled updates to the knowledge base, offline embedding generation support, and a comprehensive analytics dashboard for system monitoring and performance evaluation.

License
The project is made available for academic, research, and development use.

Acknowledgements
This system draws inspiration from modern retrieval-augmented generation designs, FAISS vector indexing techniques, the LangChain framework, and the broader communities surrounding Flask and FastAPI.


