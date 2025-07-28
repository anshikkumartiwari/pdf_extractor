# Install dependencies (minimal setup)

import json
import io
import fitz  # PyMuPDF
import re
import datetime
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

import os

input_dir = "input"
uploaded_files = {}

# Load all input files from input/ folder
for fname in os.listdir(input_dir):
    with open(os.path.join(input_dir, fname), "rb") as f:
        uploaded_files[fname] = f.read()

# Read input.json
input_json = None
for filename in uploaded_files:
    if filename.endswith(".json"):
        input_json = json.load(io.BytesIO(uploaded_files[filename]))

assert input_json is not None, "input.json not found!"

# Extract info
persona_text = input_json["persona"]["role"]
job_text = input_json["job_to_be_done"]["task"]
document_files = [doc["filename"] for doc in input_json["documents"]]

print("Persona:", persona_text)
print("Job to be done:", job_text)
print("Document files:", document_files)

# Define date patterns and month keywords for heading detection
month_keywords = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
}

date_patterns = [
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
    re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
    re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b\s+\d{1,2}', re.I),
    re.compile(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*', re.I),
]

def is_bold_or_italic(span):
    font_name = span.get("font", "").lower()
    return any(style in font_name for style in ["bold", "italic", "oblique", "slanted"])

def is_heading_candidate(span, next_span, i, merged_spans, toc_pages):
    text = span["text"].strip()
    words = text.split()
    page_num = span["page"]
    lower_text = text.lower()

    if not text or len(words) < 2 or len(words) > 15:
        return False
    if any(month in lower_text for month in month_keywords):
        return False
    if any(p.search(text) for p in date_patterns):
        return False

    above_spacing = below_spacing = True
    if page_num not in toc_pages:
        if i > 0 and merged_spans[i - 1]["page"] == span["page"]:
            above_spacing = span["y"] - merged_spans[i - 1]["y"] > 30
        if next_span and next_span["page"] == span["page"]:
            below_spacing = next_span["y"] - span["y"] > 30

    if is_bold_or_italic(span):
        return True
    return above_spacing and below_spacing

def clean_text(text):
    """Clean text by removing extra newlines, bullet points, and normalizing whitespace."""
    text = re.sub(r'\n\s*\n+', '\n', text)  # Replace multiple newlines with single
    text = re.sub(r'^\s*[\•\-\*]\s+', '', text, flags=re.MULTILINE)  # Remove bullet points
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip()

def extract_sections_from_pdf(file_bytes, filename):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    sections = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        all_spans = []
        toc_found = any("table of contents" in block.get("text", "").lower() for block in blocks)

        # Collect all spans with positional info
        for block_index, block in enumerate(blocks):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    all_spans.append({
                        "text": text,
                        "size": span["size"],
                        "font": span["font"],
                        "y": span["origin"][1],
                        "page": page_num + 1,
                        "span": span,
                        "block_index": block_index
                    })

        # Sort by vertical position
        all_spans.sort(key=lambda s: s["y"])

        # Detect headings
        heading_spans = []
        for i, span in enumerate(all_spans):
            next_span = all_spans[i + 1] if i + 1 < len(all_spans) else None
            if is_heading_candidate(span, next_span, i, all_spans, {page_num + 1} if toc_found else set()):
                heading_spans.append({
                    "text": span["text"],
                    "y": span["y"],
                    "block_index": span["block_index"]
                })

        # Extract text between headings
        for i in range(len(heading_spans)):
            heading = heading_spans[i]
            start_y = heading["y"]
            end_y = heading_spans[i + 1]["y"] if i + 1 < len(heading_spans) else None

            section_text = ""
            for block in blocks[heading["block_index"] + 1:]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        y = span["origin"][1]
                        if end_y is not None and y >= end_y:
                            break
                        if y > start_y:
                            section_text += span["text"] + " "
                    section_text += "\n"

            section_text = clean_text(section_text)
            if len(section_text) > 100:
                sections.append({
                    "document": filename,
                    "section_title": heading["text"],
                    "page_number": page_num + 1,
                    "section_text": section_text
                })

    doc.close()
    return sections

all_sections = []

for filename in document_files:
    if filename not in uploaded_files:
        print(f"Warning: {filename} not uploaded.")
        continue
    pdf_bytes = uploaded_files[filename]
    sections = extract_sections_from_pdf(pdf_bytes, filename)
    all_sections.extend(sections)

print(f"\n✅ Extracted {len(all_sections)} total section candidates.")
print("Example:\n", all_sections[:2])

# Load a more advanced model for better semantic understanding
model = SentenceTransformer('all-mpnet-base-v2')  # Better performance for semantic similarity

# Combine persona and job into one text for context
persona_job_text = f"{persona_text}. {job_text}"
persona_job_embedding = model.encode(persona_job_text, convert_to_tensor=True)

print("Persona-Job Embedding Shape:", persona_job_embedding.shape)

# Extract keywords from persona and job for boosting
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) > 3 and w not in {'with', 'from', 'that', 'this', 'your'}]

keywords = extract_keywords(persona_job_text)
print("Extracted Keywords:", keywords)

# Vectorize all section texts + persona/job as TF-IDF
section_texts = [clean_text(s['section_text'][:1000]) for s in all_sections]  # Shorten for speed
section_titles = [s['section_title'] for s in all_sections]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform([persona_job_text] + section_texts)
tfidf_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

section_scores = []

for i, section in enumerate(all_sections):
    # MiniLM similarity
    section_text = clean_text(f"{section['section_title']}. {section['section_text'][:1000]}")
    minilm_embedding = model.encode(section_text, convert_to_tensor=True)
    minilm_score = util.pytorch_cos_sim(persona_job_embedding, minilm_embedding).item()

    # TF-IDF similarity
    tfidf_score = tfidf_similarities[i]

    # Keyword boosting
    keyword_boost = 0
    for keyword in keywords:
        if keyword.lower() in section_text.lower():
            keyword_boost += 0.1  # Boost score for each matching keyword

    # Combine scores (increased weight for MiniLM for better semantic relevance)
    combined_score = 0.8 * minilm_score + 0.2 * tfidf_score + keyword_boost

    section_scores.append({
        "document": section["document"],
        "section_title": section["section_title"],
        "page_number": section["page_number"],
        "section_text": section["section_text"],
        "score": combined_score
    })

# Sort by combined score
section_scores = sorted(section_scores, key=lambda x: x["score"], reverse=True)
top_sections = section_scores[:5]

for rank, section in enumerate(top_sections, start=1):
    section["importance_rank"] = rank

# Preview
for s in top_sections:
    print(f"\n📄 {s['document']} | Rank {s['importance_rank']} | Page {s['page_number']}")
    print(f"Title: {s['section_title']}")
    print(f"Score: {s['score']:.4f}")

def extract_top_paragraphs(section, persona_job_embedding, max_chunks=2):
    paragraphs = section["section_text"].split('\n')  # Split by single newline
    paragraph_scores = []

    for para in paragraphs:
        para = clean_text(para)
        if len(para) < 100:  # Skip too short
            continue
        embedding = model.encode(para, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(persona_job_embedding, embedding).item()
        paragraph_scores.append((para, similarity))

    # Sort by score
    top_paragraphs = sorted(paragraph_scores, key=lambda x: x[1], reverse=True)[:max_chunks]
    return [p[0] for p in top_paragraphs]

output_json = {
    "metadata": {
        "input_documents": document_files,
        "persona": persona_text,
        "job_to_be_done": job_text,
        "processing_timestamp": datetime.datetime.now().isoformat()
    },
    "extracted_sections": [],
    "subsection_analysis": []
}

# Fill extracted_sections
for section in top_sections:
    output_json["extracted_sections"].append({
        "document": section["document"],
        "section_title": section["section_title"],
        "importance_rank": section["importance_rank"],
        "page_number": section["page_number"]
    })

    # Subsection analysis
    top_paras = extract_top_paragraphs(section, persona_job_embedding)
    for para in top_paras:
        output_json["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": para,
            "page_number": section["page_number"]
        })

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=4, ensure_ascii=False)

os.makedirs("output", exist_ok=True)
with open("output/output.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=4, ensure_ascii=False)
