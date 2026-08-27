# Design Document: AI Image Understanding & Content Matching Engine

## 1. Problem Statement
Automated publishing workflows often struggle to attach relevant imagery to blog posts. Relying solely on keywords or filenames frequently results in embarrassing mismatches (e.g., attaching a wolf image to an article about red foxes). This system solves the problem by combining computer vision metadata extraction, semantic vector embeddings, and a production-grade **mismatch guard** to ensure high-confidence matches and safe rejections.

## 2. Data Model Schema
Every processed image produces a validated structured metadata payload:
```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}