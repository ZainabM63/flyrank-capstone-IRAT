# AI Image Understanding & Content Matching Engine

A production-grade backend service built with **FastAPI** and **Google Gemini AI** that performs semantic image-to-post matching, vector similarity ranking, and automated safety guard evaluations to prevent content mismatches.

---

## 🚀 Key Features

* **Semantic Vector Embeddings:** Leverages Google's `models/gemini-embedding-001` model to calculate deep semantic representations of blog text and image captions.
* **Cosine Similarity Ranking:** Ranks candidate media files dynamically based on vector alignment.
* **Mismatch Guard Safety Layer:** Automated validation guard that intercepts invalid pairings (e.g., catching category mismatches like fox vs. wolf, low vision model confidence, or low semantic similarity thresholds).
* **Interactive Frontend Dashboard:** Built-in web UI served directly via FastAPI static mounts for end-to-end testing, review approvals, and rejections.
* **Audit Logging:** Records moderation actions (`APPROVE` / `REJECT`) with timestamps and audit reasons.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI, Uvicorn (Python 3.12+)
* **AI & Embeddings:** Google Generative AI SDK (`google-generativeai`)
* **Data Processing & Math:** NumPy
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

---

## 📁 Project Structure

```text
D:\IRAT\
│
├── main.py            # FastAPI application, AI endpoints, and safety guards
├── static\
│   └── index.html     # Interactive frontend testing dashboard
├── README.md          # Project documentation
└── .env               # Local environment variables (API Key configuration)