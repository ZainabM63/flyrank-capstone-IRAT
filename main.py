import os
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import google.generativeai as genai
import numpy as np
from datetime import datetime, timezone
from fastapi.staticfiles import StaticFiles

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# Configure FastAPI
app = FastAPI(
    title="AI Image Understanding & Content Matching Engine",
    version="1.0.0",
    description="Backend service for semantic image matching and safety guard filtering."
)

# Mount the static directory so FastAPI serves index.html at /static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configure Gemini API
genai.configure(api_key="your key")

# --- Pydantic Schemas for Boundary Validation ---
class ImageMetadata(BaseModel):
    subject: str
    category: str
    attributes: list[str]
    caption: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class PostRequest(BaseModel):
    post_id: str
    post_text: str

class ReviewRequest(BaseModel):
    post_id: str
    image_url: str
    action: str  # "APPROVE" or "REJECT"
    reason: str | None = None

# --- In-Memory Mock Database for Corpus ---
DATABASE = {
    "images": [
        {
            "image_url": "https://picsum.photos/seed/fox/400/300",
            "metadata": {
                "subject": "red fox",
                "category": "animal",
                "attributes": ["orange fur", "wild", "forest"],
                "caption": "A red fox standing alert in a green forest",
                "confidence": 0.94
            }
        },
        {
            "image_url": "https://picsum.photos/seed/wolf/400/300",
            "metadata": {
                "subject": "gray wolf",
                "category": "animal",
                "attributes": ["gray fur", "wild", "snow"],
                "caption": "A gray wolf prowling through a snowy pine forest",
                "confidence": 0.91
            }
        }
    ],
    "posts": [
        {
            "post_id": "post-1",
            "post_text": "The behavior, diet, and habitat of Vulpes vulpes (the red fox) in the wild."
        }
    ],
    "reviews": []
    }

# --- Core AI & Matching Logic ---
def get_embedding(text: str) -> list[float]:
    """Generate vector embedding using Gemini's embedding model."""
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    return response['embedding']

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def mismatch_guard(post_text: str, image_metadata: dict, similarity_score: float, min_similarity: float = 0.70, min_confidence: float = 0.80) -> dict:
    """
    The production safety layer: decides if a recommendation is good enough
    or needs a safe rejection.
    """
    # Check 1: Vision Model Confidence
    if image_metadata.get("confidence", 0.0) < min_confidence:
        return {
            "status": "REJECTED",
            "reason": f"Low model confidence: {image_metadata.get('confidence')} is below threshold {min_confidence}"
        }

    # Check 2: Similarity Threshold
    if similarity_score < min_similarity:
        return {
            "status": "REJECTED",
            "reason": f"Semantic similarity too low: {similarity_score:.2f} is below threshold {min_similarity}"
        }

    # Check 3: Category / Concept Sanity Guard (Fox vs Wolf protection)
    post_lower = post_text.lower()
    subject = image_metadata.get("subject", "").lower()
    
    if "fox" in post_lower and "wolf" in subject:
        return {
            "status": "REJECTED",
            "reason": "Animal category mismatch: expected fox content, detected wolf image."
        }
    
    if "wolf" in post_lower and "fox" in subject:
        return {
            "status": "REJECTED",
            "reason": "Animal category mismatch: expected wolf content, detected fox image."
        }

    return {
        "status": "APPROVED",
        "reason": f"Passed all safety checks with similarity {similarity_score:.2f} and confidence {image_metadata.get('confidence')}."
    }

# --- API Endpoints ---
@app.get("/posts/{post_id}/images")
def get_post_image_suggestions(post_id: str):
    """Retrieve ranked, mismatch-guarded image suggestions for a blog post."""
    # Find post
    post = next((p for p in DATABASE["posts"] if p["post_id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Blog post not found.")

    post_vector = get_embedding(post["post_text"])
    suggestions = []

    for img in DATABASE["images"]:
        caption = img["metadata"]["caption"]
        img_vector = get_embedding(caption)
        similarity = cosine_similarity(post_vector, img_vector)

        # Run through the Mismatch Guard safety layer
        guard_result = mismatch_guard(post["post_text"], img["metadata"], similarity)

        suggestions.append({
            "image_url": img["image_url"],
            "caption": caption,
            "similarity_score": round(similarity, 4),
            "guard_evaluation": guard_result
        })

    # Sort candidates by similarity score descending
    suggestions = sorted(suggestions, key=lambda x: x["similarity_score"], reverse=True)

    return {
        "post_id": post_id,
        "suggestions": suggestions
    }

@app.post("/reviews")
def review_pairing(review: ReviewRequest):
    """Approve or reject a suggested image pairing with an audit reason."""
    review_record = {
        "post_id": review.post_id,
        "image_url": review.image_url,
        "action": review.action.upper(),
        "reason": review.reason,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    DATABASE["reviews"].append(review_record)
    return {"message": "Review recorded successfully", "record": review_record}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "image-matching-engine"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)