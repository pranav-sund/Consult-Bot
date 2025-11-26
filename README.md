
Consultancy & Advisory – Intelligent Chatbot System

A Hybrid Rule-Based, Semantic Search, and AI-Enhanced Advisory Assistant

 Overview

The Consultancy & Advisory Chatbot System is a complete conversational platform designed to automate client interactions in consultancy firms. It handles:

Real-time FAQ answering

Service & pricing queries

Appointment scheduling

Chat logging

PDF summary generation

Admin FAQ management

The system is primarily built using Python, Flask, and SQLite, with a modular architecture ready for future scaling into an AI-powered advisory system using FastAPI, LangChain, FAISS, and Embeddings.

This README documents both:

 Current Production System (Flask-based) — dominant
 Future AI-Enhanced System (FastAPI + Semantic Search) — integrated extension
 Core Features (Current System – Flask + SQLite)
 Client Capabilities

Client registration & secure login

Real-time rule-based chatbot

FAQ responses (services, pricing, availability)

Appointment booking

Email notifications

PDF download of chat summary

 Admin Capabilities

Protected admin login

Update FAQs dynamically

View chat logs & booked appointments

 System Features

Rule-based chatbot engine

SQLite database

Secure password hashing

PDF report generation (ReportLab)

Email service (Flask-Mail)

Modular, scalable architecture

 Future System Expansion – AI, Embeddings & Enhanced Logic

In the next iteration, this project evolves into a full AI-driven advisory assistant using:

 Mode A – Rule-Based FAQ Engine

Fastest

Offline

Deterministic responses

CSV-based lookup

 Mode B – Semantic Search (RAG)

Embeddings (OpenAI/local models)

FAISS vector store

Natural-language understanding

Powered by FastAPI + LangChain

 Mode C – Full Advisory Assistant

Hybrid (Rule + Embedding)

Appointment booking

Chat logging into SQLite

PDF transcript export

Reloadable knowledge base (CSV)

Dataset Information (Future AI System)

The file Chatbot_System_Merged_30Rows.csv includes:

Column	Description
Question	FAQ question text
Answer	Predefined chatbot answer
Client_ID	Unique client identifier
Query	Logged user queries
Response	Chatbot-generated replies
Appointment_ID	Appointment entry reference
Date, Time	Booking slots
Admin_ID	Admin-level access
Admin_Password	Admin authentication

Used primarily for FAQ lookup + semantic search.

System Architecture
Current (Production) Architecture – Flask Version

Three-tier architecture:

1 Presentation Layer

HTML, CSS, Bootstrap, Jinja2 templates.

2 Application Layer

Flask backend handles:

Chat logic

Login & authentication

FAQ/DB queries

Appointments

Email & PDF generation

3️. Data Layer

SQLite database for:

Clients

FAQs

Chat logs

Appointments

Admin credentials

Future AI Architecture – FastAPI + FAISS + LangChain
+-----------------------------+
|       User Interface        |
+-------------+---------------+
              |
              v
+-----------------------------+
|        FastAPI Server       |
|         /consultbot         |
+-------------+---------------+
              |
-------------------------------------------------------------
|                         |                                 |
v                         v                                 v
Rule-Based Engine   CSV Knowledge Base        Embedding Engine
(fast lookup)       (30-row dataset)        (FAISS + LangChain)
              |
              v
+-----------------------------+
|    Appointment Manager      |
|           SQLite            |
+-------------+---------------+
              |
              v
+-----------------------------+
|   Chat Logs & PDF Engine    |
|         ReportLab           |
+-----------------------------+

 Project Structure (Unified)
chatbot_project/
├── app/                         # Flask backend (current system)
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── templates/
│   └── static/
├── consultbot_app.py            # Future FastAPI AI engine
├── Chatbot_System_Merged_30Rows.csv
├── consultbot_data/
│   ├── consultbot.db
│   ├── faiss_index/
│   ├── PDFs/
├── tests/
├── migrations/
├── requirements.txt
└── README.md

 Tech Stack
Current Implementation

Python 3.8+

Flask

SQLite + SQLAlchemy

ReportLab (PDF)

Flask-Mail

Bootstrap 5

Werkzeug / bcrypt

Future AI Implementation

FastAPI

FAISS

LangChain

OpenAI / Local Embedding Models

Pandas

Uvicorn

 Testing Summary

 Unit tests for chatbot logic, DB operations, appointments
 Integration tests for chat → DB → email/PDF flow
 Security tests (hashed passwords, SQL injection prevention, CSRF)
 System tests across browsers

All major modules passed successfully.

 Security Measures

Password hashing (bcrypt/werkzeug)

CSRF protection

Role-based access control

Secure cookies (HttpOnly, SameSite)

HTTPS (Flask-Talisman, deployment stage)

SQL injection protection via ORM

Input validation

 Installation & Setup

1 Clone
git clone <your-repo-url>
cd chatbot_project

2️. Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac

3️. Install Packages
pip install -r requirements.txt

4️. Place Dataset (Future AI System)
Chatbot_System_Merged_30Rows.csv

5️. Run Flask System
python run.py


Visit: http://localhost:5000

6️. Run Future FastAPI System
uvicorn consultbot_app:app --reload

 Future API Endpoints (FastAPI System)

POST /consultbot – Main chatbot endpoint

POST /appointments/create – Create appointment

GET /chatlogs – Retrieve logs

POST /chat_summary_pdf – Generate PDF

POST /admin/reload_kb – Reload CSV KB

 Future Enhancements

Full AI advisory system

Multilingual support

Voice chatbot interface

Web/React-based frontend

Admin authentication for KB reload

Offline embedding generation

Advanced analytics dashboard

 License

This project is available for academic, research, and development use.

 Acknowledgements

Inspired by:

Modern RAG architectures

FAISS vector search

LangChain ecosystem

Flask & FastAPI communities

