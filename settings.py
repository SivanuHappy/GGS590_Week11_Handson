"""Settings shared by every module: paths, models, the fact sheets and limits. Change values here."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / "project_data_analysis_agent" / ".env")   # reuse the key of the air-quality project

# Groq model used by all agents. If Groq retires it, change it here (or set GROQ_MODEL).
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Second model, used only when the first one fails (for example when it garbles a tool call). "" turns it off.
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")

DOCS_DIR = BASE_DIR / "docs"
SKILLS_DIR = BASE_DIR / "skills"

# The fact sheets: a short key (the agents use it), the PDF in docs/, and the title shown in answers.
SHEETS = {
    "before_smoke": ("prepare-for-wildfire-smoke-factsheet.pdf", "Prepare for Wildfire Smoke"),
    "while_smoke": ("reduce-smoke-exposure-factsheet.pdf", "Reduce Smoke Exposure Indoors"),
    "after_smoke": ("protect-yourself-from-ash-factsheet.pdf", "Protect Yourself from Ash"),
}

# Search: the PDFs are cut into short pieces, and each piece is turned into numbers (an embedding) by a small
# local model, so a question can be matched by meaning and not only by shared words.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600         # characters per piece
CHUNK_OVERLAP = 100      # characters shared by neighbouring pieces
TOP_K = 4                # most pieces one search returns
MIN_SCORE = 0.25         # a piece scoring below this (cosine similarity, 0 to 1) is not a real match

# Limits.
MAX_TOOL_CALLS = 10      # tool calls per question, shared by the supervisor and all specialists
KEEP_QUESTIONS = 3       # how many recent questions the supervisor remembers
