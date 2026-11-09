"""
The fact sheets (the retrieval part of RAG) and the tools the agents use.

Load:    load_guides() reads the PDFs, cuts them into short pieces, embeds them and keeps them in memory.
Tools:   search_guidance (finds the best matching pieces), load_skill (loads a checklist from skills/)
The model only chooses a tool and its options; the pieces it reads always come from the PDFs in docs/.
"""
import json
import re
import warnings
from typing import Literal

import pypdf
import yaml
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import settings

store = None          # the in-memory vector store; filled by load_guides()
chunk_count = 0
sheet_pages = {}      # sheet key -> number of pages


# ---------------------------------------------------------------------------
# Loading the fact sheets: PDF -> pages -> short pieces -> embeddings in memory
# ---------------------------------------------------------------------------
# Lines of the "Resources" boxes that are only link names; they would match many questions without answering any.
LINK_LINES = re.compile(r"^[-•o]?\s*(AirNow Fire and Smoke Map|IOS App|Android App|Wildfire Guide Fact Sheets|"
                        r"Resources and Information on Wildland Fires|Infographics on Staying Safe|Scan the QR code|"
                        r"Resources$|Additional Resources$)", re.IGNORECASE)


def _clean(text: str) -> str:
    """Tidy text taken from a PDF: bullet glyphs, repeated spaces, empty lines and link-name lines."""
    text = re.sub(r"[-]", "-", text)          # private-use bullet glyphs
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line and not LINK_LINES.match(line))


def load_guides():
    """Read every PDF page by page, split it into pieces, embed the pieces and keep them in memory."""
    global store, chunk_count
    pages = []
    for key, (filename, title) in settings.SHEETS.items():
        path = settings.DOCS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Fact sheet not found: {path}. Put the PDF in the docs/ folder.")
        reader = pypdf.PdfReader(path)
        sheet_pages[key] = len(reader.pages)
        for number, page in enumerate(reader.pages, start=1):
            pages.append(Document(page_content=_clean(page.extract_text() or ""),
                                  metadata={"sheet": key, "title": title, "page": number}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", ". ", " "])
    pieces = [piece for piece in splitter.split_documents(pages) if len(piece.page_content) > 40]
    store = InMemoryVectorStore(HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL))
    store.add_documents(pieces)
    chunk_count = len(pieces)
    return store


# ---------------------------------------------------------------------------
# Checking citations against the real sheets (plain code, no model)
# ---------------------------------------------------------------------------
GROUP = re.compile(r"\(([^()]*?\bpp?\.\s*\d[^()]*)\)")   # text in brackets that has a page: (Sheet, p. 2)
PART = re.compile(r"^\s*[\"“]?(.+?)[\"”]?\s*,\s*pp?\.\s*([\d,\s–\-]+?)\s*$")   # one "Sheet title, p. N"


def check_citations(text: str) -> list:
    """Return warnings for citations to a sheet that does not exist or a page it does not have.
    An empty list means every citation points to a real sheet and page."""
    titles = {title.lower(): key for key, (_, title) in settings.SHEETS.items()}
    warnings = []
    for group in GROUP.findall(text):
        for part in group.split(";"):                                # (Sheet A, p. 1; Sheet B, p. 2) holds two citations
            match = PART.match(part)
            if not match:
                continue
            title = match.group(1).strip(" \"“”")
            key = titles.get(title.lower())
            if key is None:
                warnings.append(f'Unknown fact sheet in a citation: "{title}"')
                continue
            for number in map(int, re.findall(r"\d+", match.group(2))):
                if not 1 <= number <= sheet_pages.get(key, 0):
                    warnings.append(f'"{title}" has no page {number} (it has {sheet_pages.get(key, 0)}).')
    return list(dict.fromkeys(warnings))


# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------
@tool
def search_guidance(query: str, sheet: Literal["any", "before_smoke", "while_smoke", "after_smoke"] = "any") -> str:
    """
    Search the EPA wildfire smoke fact sheets and return the best matching passages with their sheet and page.
    query: the question or topic in plain words, for example 'keep smoke out with central air conditioning'.
    sheet: any (all three sheets), before_smoke (Prepare for Wildfire Smoke: before smoke arrives),
      while_smoke (Reduce Smoke Exposure Indoors: during smoke, indoors and outdoors),
      after_smoke (Protect Yourself from Ash: after the fire, ash and cleanup).
    If nothing in the sheets matches, it says so: do not answer from memory then.
    """
    if store is None:
        return json.dumps({"error": "the fact sheets are not loaded"})
    if not query.strip():
        return json.dumps({"error": "give a query"})
    keep = None if sheet == "any" else (lambda doc: doc.metadata.get("sheet") == sheet)
    results = store.similarity_search_with_score(query, k=settings.TOP_K, filter=keep)
    matches = [{"sheet": doc.metadata["title"], "page": doc.metadata["page"], "score": round(float(score), 2),
                "text": doc.page_content}
               for doc, score in results if score >= settings.MIN_SCORE]
    if not matches:
        return json.dumps({"matches": [], "note": "No passage in the fact sheets matches this query. Try different "
                           "words once, and if there is still no match, say the sheets do not cover it."})
    return json.dumps({"matches": matches})


# ---------------------------------------------------------------------------
# Skills: checklist files in the skills/ folder, loaded only when a question needs them
# ---------------------------------------------------------------------------
def _read_skills() -> dict:
    """Load the Markdown skills; each file has a YAML header with a name and a description."""
    skills = {}
    for path in sorted(settings.SKILLS_DIR.glob("*.md")):
        parts = path.read_text(encoding="utf-8").split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            warnings.warn(f"Skipped {path.name}: expected a YAML header between --- lines.")
            continue
        try:
            meta = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            warnings.warn(f"Skipped {path.name}: invalid YAML header.")
            continue
        if not isinstance(meta, dict) or not all(isinstance(meta.get(k), str) for k in ("name", "description")):
            warnings.warn(f"Skipped {path.name}: name and description must be text.")
            continue
        skills[meta["name"]] = {"description": meta["description"], "body": parts[2].strip()}
    return skills


SKILLS = _read_skills()


@tool
def load_skill(name: str) -> str:
    """
    Load the step-by-step checklist of a skill by its name. Call this first when a question
    matches one of the available skills listed in the system prompt, then follow the steps.
    """
    if name not in SKILLS:
        return json.dumps({"error": f"unknown skill '{name}'. Skills are: {sorted(SKILLS)}"})
    return SKILLS[name]["body"]


# ---------------------------------------------------------------------------
# Which agent gets which tools
# ---------------------------------------------------------------------------
SPECIALIST_TOOLS = [search_guidance, load_skill]
REVIEWER_TOOLS = [search_guidance]
