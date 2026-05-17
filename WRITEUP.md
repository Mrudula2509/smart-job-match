# WRITEUP.md

# Smart Job Match Agent – Technical Write-up

## 1. Design Choices

For semantic job matching, I selected the Cohere Embeddings API (`embed-english-v3.0`) as the embedding model. The main reason for this decision was deployment efficiency and scalability. Initially, I experimented with local SentenceTransformers models such as `all-MiniLM-L6-v2`, which produced good semantic similarity results locally. However, these local transformer models significantly increased the project bundle size and exceeded Vercel’s serverless deployment limits.

Using Cohere’s hosted embedding API solved this problem because embeddings were generated externally instead of loading a large transformer model into memory. This reduced deployment complexity and allowed the backend to run efficiently on Vercel’s free serverless environment.

I also considered using TF-IDF vectorization as a fallback lightweight embedding approach. While TF-IDF is fast and deployable, it lacks strong semantic understanding compared to transformer embeddings. It performs poorly when resumes and job descriptions use different wording for similar concepts. Therefore, I rejected TF-IDF as the primary embedding system and kept semantic embeddings as the core ranking method.

The main trade-off I made was relying on external APIs instead of fully local inference. This introduces dependency on internet connectivity and API rate limits, but it provided significantly better semantic quality while keeping deployment lightweight and practical.

For reasoning and question generation, I used Groq-hosted Llama models. Groq provided fast inference speeds and low latency for generating explanations, follow-up questions, and refinement reasoning.

---

## 2. Agentic Architecture

The system uses a lightweight multi-step agentic workflow instead of one large prompt.

The architecture is divided into two primary tool calls:

1. Resume Parsing Tool
2. Match Reasoning Tool

The first tool extracts structured candidate information such as:
- skills
- experience
- preferred roles
- education

This structured information is then passed into the second reasoning tool, which:
- explains why jobs match
- generates clarifying questions
- performs ranking refinement

I intentionally separated the workflow into multiple stages because combining everything into one large prompt caused inconsistent and difficult-to-control outputs during testing. Splitting the workflow improved modularity, debugging, and reasoning quality.

The ranking pipeline works as follows:

1. Job descriptions are converted into embeddings during startup.
2. The resume text is converted into embeddings at request time.
3. Cosine similarity is used to rank jobs.
4. The top-ranked jobs are passed to the reasoning agent.
5. The reasoning agent generates explanations and a clarifying question.
6. Candidate answers are later used in `/refine` to update rankings.

The main failure modes of this architecture are:
- incorrect resume parsing
- hallucinated reasoning
- weak clarifying questions
- embedding mismatch for unusual resumes

If the LLM fails during parsing or reasoning, fallback logic is used to keep the API functional instead of crashing.

---

## 3. Honest Weaknesses

The system still has several limitations.

Poorly written resumes can reduce ranking quality significantly. For example:
- missing punctuation
- incomplete sentences
- excessive spelling mistakes
- highly unstructured resumes

can reduce embedding quality and cause inaccurate matching.

Another weakness is that the clarifying question generation can occasionally become repetitive if many top jobs share similar characteristics. During testing, the model sometimes repeatedly asked about remote versus onsite preferences because many jobs in the dataset emphasized remote work.

At large scale, the system would face multiple bottlenecks. With 10,000 concurrent requests:
- external API rate limits would become a problem
- serverless cold starts would increase latency
- repeated embedding generation would become expensive

Currently, embeddings are generated per request without caching, which would not scale efficiently for production workloads.

Due to time limitations, I also cut several corners:
- no authentication
- no database persistence
- no caching layer
- no async queue system
- no advanced prompt optimization
- no retry handling for API failures

The current system is optimized primarily for assignment functionality and deployment simplicity rather than enterprise-scale reliability.

---

## 4. Next Steps

If I had two additional days, the single highest-impact improvement would be implementing a retrieval and caching layer for embeddings and ranking.

Currently, embeddings are generated repeatedly for every request. By introducing:
- Redis caching
- vector storage
- precomputed resume embeddings
- approximate nearest neighbor search

the system could scale much more efficiently while reducing latency and API costs.

I would also improve the refinement pipeline by incorporating weighted candidate preferences into ranking instead of simply appending answers to the resume text. This would produce more stable and explainable ranking updates.

Additionally, I would improve robustness for noisy resumes using preprocessing steps such as:
- spell correction
- section detection
- resume normalization

These improvements would make the system significantly more production-ready while preserving the current agentic workflow.
