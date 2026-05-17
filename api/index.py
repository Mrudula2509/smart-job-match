import os
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cohere
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

# =========================
# API Configuration
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env")

# =========================
# Groq Client
# =========================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "llama-3.1-8b-instant"

# =========================
# Cohere Client
# =========================

co = cohere.Client(COHERE_API_KEY)

# =========================
# Load Dataset
# =========================

_here = Path(__file__).resolve().parent

for candidate in [_here / "jobs.json", _here.parent / "jobs.json"]:
    if candidate.exists():
        JOBS_PATH = candidate
        break
else:
    raise FileNotFoundError("jobs.json not found")

with open(JOBS_PATH, encoding="utf-8") as f:
    JOBS = json.load(f)

# =========================
# Embedding Setup
# =========================

_job_embeddings = []


def embed_texts(texts):

    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )

    return response.embeddings


def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return round(dot / (norm_a * norm_b), 6)


def job_to_text(job):

    skills = ", ".join(job.get("skills", []))

    return (
        f"{job['title']} at {job['company']} "
        f"in {job['domain']}. "
        f"Skills: {skills}. "
        f"Experience: {job['experience_years']} years. "
        f"{job['description']}"
    )


def rank_jobs(resume_text, top_n=5):

    resume_embedding = embed_texts([resume_text])[0]

    scored = []

    for job, job_embedding in zip(JOBS, _job_embeddings):

        score = cosine_similarity(
            resume_embedding,
            job_embedding
        )

        scored.append({
            **job,
            "similarity_score": score
        })

    scored.sort(
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    return scored[:top_n]

# =========================
# Startup
# =========================


@asynccontextmanager
async def lifespan(app: FastAPI):

    global _job_embeddings

    print("Generating job embeddings...")

    _job_embeddings = embed_texts(
        [job_to_text(job) for job in JOBS]
    )

    print("Embeddings ready.")

    yield

# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Smart Job Match Agent",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helper Functions
# =========================


def extract_name(resume_text):

    lines = resume_text.strip().split("\n")

    if lines:
        return lines[0].strip()

    return "Candidate"

# =========================
# TOOL 1 — Resume Parser
# =========================


def resume_parser_tool(resume_text):

    prompt = f"""
Extract:
- candidate name
- skills
- years of experience
- preferred roles
- education

Return ONLY valid JSON.

Resume:
{resume_text}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            #response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        parsed = json.loads(
            response.choices[0].message.content
        )

        return parsed

    except Exception as e:

        print("RESUME PARSER ERROR:", e)

        return {
            "name": extract_name(resume_text),
            "skills": [],
            "experience_years": 1,
            "preferred_roles": [],
            "education": "Bachelor's Degree"
        }

# =========================
# TOOL 2 — Match Reasoning
# =========================


def match_reasoning_tool(candidate_data, top_jobs):

    explanations = []

    for job in top_jobs:

        prompt = f"""
Candidate:
{candidate_data}

Job:
{job['title']} at {job['company']}

Required Skills:
{job['skills']}

Explain in 2 short sentences why this job matches the candidate.
"""

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5
            )

            explanation = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

        except Exception:

            explanation = (
                f"This role matches skills like "
                f"{', '.join(job['skills'][:3])}."
            )

        explanations.append({
            "job_id": job["id"],
            "explanation": explanation
        })

    # =========================
    # Dynamic Question
    # =========================

    question_prompt = f"""
Candidate:
{candidate_data}

Top Jobs:
{[job['title'] for job in top_jobs[:3]]}

Generate ONE short smart follow-up question.

Rules:
- Must depend on candidate profile
- Must depend on matched jobs
- Avoid generic questions
- Avoid repeating remote/on-site questions
- Return ONLY plain text
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": question_prompt
                }
            ],
            temperature=0.9
        )

        question = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        # =========================
        # Block repetitive question
        # =========================

        if (
            "remote" in question.lower()
            or "onsite" in question.lower()
            or "on-site" in question.lower()
        ):

            domains = list(set(
                [job["domain"] for job in top_jobs]
            ))

            question = (
                f"You have strong matches across "
                f"{', '.join(domains)} domains. "
                f"Which industry interests you most?"
            )

    except Exception:

        question = (
            "Which type of roles interests you most?"
        )

    return {
        "explanations": explanations,
        "clarifying_question": question
    }

# =========================
# AGENT ORCHESTRATOR
# =========================


def run_agent(resume_text, top_jobs):

    parsed_candidate = resume_parser_tool(
        resume_text
    )

    reasoning_result = match_reasoning_tool(
        parsed_candidate,
        top_jobs
    )

    return {
        "parsed_candidate": parsed_candidate,
        "reasoning_result": reasoning_result
    }

# =========================
# Request Models
# =========================


class RecommendRequest(BaseModel):

    resume_text: str

    @field_validator("resume_text")
    @classmethod
    def validate_resume(cls, v):

        v = v.strip()

        if not v:
            raise ValueError(
                "resume_text cannot be empty"
            )

        if len(v) < 50:
            raise ValueError(
                "Resume text too short"
            )

        return v


class RefineRequest(BaseModel):

    resume_text: str
    clarifying_question: str
    candidate_answer: str

    @field_validator(
        "resume_text",
        "clarifying_question",
        "candidate_answer"
    )
    @classmethod
    def validate_fields(cls, v):

        v = v.strip()

        if not v:
            raise ValueError(
                "Field cannot be empty"
            )

        return v

# =========================
# Routes
# =========================


@app.get("/")
def root():

    return {
        "service": "Smart Job Match Agent",
        "jobs_loaded": len(JOBS),
        "status": "running"
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):

    start_time = time.time()

    top_jobs = rank_jobs(req.resume_text)

    agent_output = run_agent(
        req.resume_text,
        top_jobs
    )

    parsed = agent_output["parsed_candidate"]
    reasoning = agent_output["reasoning_result"]

    explanation_map = {
        e["job_id"]: e["explanation"]
        for e in reasoning.get("explanations", [])
    }

    return {
        "candidate": {
            "name": parsed.get("name", ""),
            "skills": parsed.get("skills", []),
            "experience_years":
                parsed.get("experience_years", 0),
            "preferred_roles":
                parsed.get("preferred_roles", []),
            "education":
                parsed.get("education", "")
        },

        "ranked_jobs": [
            {
                "id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "remote": job["remote"],
                "domain": job["domain"],
                "skills": job["skills"],
                "salary_lpa": job["salary_lpa"],
                "similarity_score":
                    job["similarity_score"],
                "explanation":
                    explanation_map.get(
                        job["id"],
                        "No explanation generated."
                    )
            }
            for job in top_jobs
        ],

        "clarifying_question":
            reasoning.get(
                "clarifying_question",
                ""
            ),

        "processing_time_seconds":
            round(time.time() - start_time, 2)
    }


@app.post("/refine")
def refine(req: RefineRequest):

    augmented_resume = (
        f"{req.resume_text}\n\n"
        f"Question: {req.clarifying_question}\n"
        f"Answer: {req.candidate_answer}"
    )

    top_jobs = rank_jobs(
        augmented_resume
    )

    prompt = f"""
A candidate answered a clarifying question.

Resume:
{req.resume_text[:500]}

Question:
{req.clarifying_question}

Answer:
{req.candidate_answer}

Updated top jobs:
{json.dumps(top_jobs, indent=2)}

Explain clearly how the candidate's latest answer affected the rankings.
Mention which preferences, skills, or constraints caused certain jobs
to move higher or lower.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.6
        )

        reasoning = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception:

        reasoning = (
            "The rankings changed based "
            "on the candidate's updated "
            "preferences."
        )

    return {
        "ranked_jobs": [
            {
                "id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "similarity_score":
                    job["similarity_score"]
            }
            for job in top_jobs
        ],

        "reasoning": reasoning
    }
