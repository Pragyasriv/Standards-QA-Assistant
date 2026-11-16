
# Project: Standards Q&A Assistant – Citation-Backed RAG

## Project Objective
Build a production-quality Retrieval-Augmented Generation (RAG) application that answers questions using official RoHS and REACH regulatory documents, with clause-level citations.

## Technology Stack
- Python 3.11
- FastAPI for REST APIs
- LangChain for RAG orchestration
- Qdrant for vector storage and retrieval
- Sentence Transformers for local embeddings
- Groq API for LLM inference
- RAGAS for RAG evaluation
- Pytest for automated testing

## Architecture
Build separate modules for:
1. Document ingestion
2. Text extraction and cleaning
3. Structure-aware chunking
4. Embedding generation
5. Qdrant storage
6. Semantic retrieval
7. Domain-specific guardrails
8. LLM-based answer generation
9. Citation generation
10. FastAPI endpoints
11. RAGAS evaluation

## Coding Rules
- Use clean, modular, maintainable Python code.
- Follow PEP 8 conventions.
- Use type hints and meaningful docstrings.
- Keep business logic separate from API routes.
- Never hardcode API keys or secrets.
- Read configuration from environment variables.
- Use Pydantic for request and response validation.
- Handle exceptions properly.
- Add logging instead of relying on print statements.
- Write unit tests for important components.
- Avoid unnecessary abstractions and overengineering.
- Do not generate fake regulatory documents or citations.
- Never invent article numbers or clause references.
- Preserve document metadata throughout the pipeline.

## Development Rules
- Implement the project phase by phase.
- Do not implement future phases without explicit approval.
- Before coding, inspect the existing project structure.
- Explain the implementation plan before making significant changes.
- After implementation, run relevant tests.
- Fix errors before declaring a phase complete.
- Do not silently skip failed tests.
- Provide a summary of files created or modified.
- Document commands required to run and test the code.

## Reliability Requirements
- Answers must be grounded in retrieved regulatory context.
- Unsupported questions must be rejected.
- Every factual answer must contain source citations.
- If evidence is insufficient, explicitly say so.
- Never present generated legal interpretations as definitive legal advice.
- Distinguish between retrieved evidence and generated explanations.

## Evaluation Requirements
- Maintain a manually curated evaluation dataset.
- Evaluate answer faithfulness and retrieval quality.
- Measure off-topic rejection performance separately.
- Report actual measured results only.
- Never fabricate evaluation scores.

## Git Rules
- Never commit secrets or .env files.
- Use meaningful commit messages.
- Keep generated datasets and virtual environments out of Git.
