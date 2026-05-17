# Smart Job Match Agent

An AI-powered backend system that intelligently matches candidate resumes with relevant job opportunities using semantic embeddings, cosine similarity, and LLM-based reasoning.

The system also generates dynamic follow-up questions to resolve ambiguities and supports re-ranking job recommendations based on candidate responses.

---

# Features

- Semantic job matching using embeddings
- Cosine similarity ranking
- Dynamic AI-generated clarifying questions
- AI-powered reasoning for recommendations
- Refinement endpoint for updated rankings
- FastAPI backend
- Swagger API documentation
- Public deployment on Vercel

---

# Tech Stack

- Python
- FastAPI
- Cohere Embeddings API
- Groq LLM API
- NumPy
- Vercel

---

# Live Deployment

https://smart-job-match-gules.vercel.app/docs

---

# Project Structure

```text
SMART-JOB-MATCH/
│
├── api/
│   └── index.py
│
├── jobs.json
├── requirements.txt
├── vercel.json
├── .env.example
├── .gitignore
└── README.md
```

---

# API Endpoints

## GET /

Health check endpoint.

### Example Response

```json
{
  "service": "Smart Job Match Agent",
  "jobs_loaded": 50,
  "status": "running"
}
```

---

## POST /recommend

Matches a resume with the most relevant jobs.

### Example Input

```json
{
  "resume_text": "Backend Developer with Python, FastAPI, Docker, PostgreSQL, and AWS experience."
}
```

### Output

- Ranked job recommendations
- Similarity scores
- AI-generated explanations
- Dynamic clarifying question

---

## POST /refine

Re-ranks jobs after receiving the candidate's answer to the clarifying question.

### Example Input

```json
{
  "resume_text": "Backend Developer with Python and FastAPI experience.",
  "clarifying_question": "Which backend domain interests you most?",
  "candidate_answer": "I am interested in scalable distributed backend systems."
}
```

### Output

- Updated rankings
- AI-generated reasoning explaining ranking changes

---

# Local Setup

## Clone Repository

```bash
git clone <your-github-repo-url>
cd smart-job-match
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Create Environment File

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

## Run Server

```bash
uvicorn api.index:app --reload
```

## Open Swagger Docs

```text
http://127.0.0.1:8000/docs
```

---

# Environment Variables

| Variable | Description |
|---|---|
| GROQ_API_KEY | Groq API key for LLM reasoning |
| COHERE_API_KEY | Cohere API key for embeddings |

---

# Embedding & Ranking Logic

1. Resume text is converted into embeddings using Cohere.
2. Job descriptions are pre-embedded during startup.
3. Cosine similarity is used to rank jobs.
4. Groq LLM generates:
   - recommendation reasoning
   - clarifying questions
   - refinement explanations

---

# Deployment

The project is deployed on Vercel using FastAPI serverless functions.

---

# Notes

- Real API keys are NOT included in this repository.
- Use `.env.example` as a template.
- Designed for internship assignment evaluation and AI backend demonstration.

---

# Author

Mrudula Rewatkar
