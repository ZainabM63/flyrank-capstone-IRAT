# Build Log - AI Image Understanding & Content Matching Engine

## Phase 1: Design & Architecture
- Created design document (`DESIGN.md`) outlining the image metadata schema, vector embedding storage, and mismatch guard rules.
- Set up project directory structure separating data ingestion, semantic matching, safety guards, and HTTP routing.

## Phase 2: Image Understanding Pipeline
- Integrated Gemini Flash vision model to process images from the local corpus (~50 images).
- Implemented structured output validation to ensure metadata conforms strictly to the expected JSON schema (subject, category, attributes, caption, confidence score).
- Added retry logic and per-call cost tracking for bulk processing[cite: 2].

## Phase 3: Matching & The Mismatch Guard
- Implemented vector embeddings generation for both image captions and blog post texts using Gemini embedding models
- Created cosine similarity ranking algorithms to score and retrieve candidate images.
- Built the production safety layer (**mismatch guard**) to intercept low-confidence classifications, low similarity scores, and conceptual category mismatches (e.g., wolf-on-a-fox-post)[cite: 2].

## Phase 4: Production Layer & Evaluation
- Built review and inspection API endpoints to approve or reject pairings.
- Created a small labeled evaluation dataset to calculate top-1 precision
- Documented proofs and run instructions in `EVIDENCE.md` and `README.md`.