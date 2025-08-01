# 🧠 Document Section Extractor and Relevance Scorer

A powerful, fully offline tool that **extracts sections from PDF documents** and ranks them by **relevance to a specific task or job role**. Ideal for resume screening, itinerary generation, educational content parsing, and more.

> 📍 Built with semantic embeddings, ONNX optimization, and TF-IDF scoring for precision and speed.

---

## 🚀 Features

👉 **Offline execution**
👉 **CPU-only inference** (ONNX)
👉 **Model size ≤ 500MB**
👉 **< 60 seconds** for 3–5 PDFs

### 🔍 Intelligent Section Extraction

* Headings detected via:

  * **Font style** (bold, italic, oblique)
  * **Vertical spacing** > 30 units
  * **Exclusion of dates/months** using regex
  * **Title-like spans** (2–15 words)

### 🌀 Relevance Ranking

* **Semantic Embeddings**: `all-mpnet-base-v2` via ONNX
* **TF-IDF Boosting** for keyword alignment
* **Combined scoring**: `0.8 * Embedding Score + 0.2 * TF-IDF + Keyword Boost`

### 📼 Output Format

* JSON with:

  * Ranked sections
  * Key paragraph highlights
  * Metadata (timestamp, persona, documents)

---

## 🧠 Core Techniques & Methodology (From Source Code)

### 🔖 PDF Section Extraction

* PyMuPDF used to extract structured blocks/spans
* Blocks filtered based on vertical layout, spacing, and styling
* Removes common pitfalls (dates, months, too short titles)
* Sections extracted as spans between detected headings

### 🧬 Text Cleaning

* Bullet and newline normalization
* Removal of excessive whitespace and formatting

### 📊 Semantic Relevance Scoring

* Embedding model: `all-mpnet-base-v2`
* Embeddings generated for:

  * Persona + job context
  * Section content (title + top 1000 chars)
* Cosine similarity used for semantic matching (via `sentence-transformers` + `torch`)

### 🔢 TF-IDF Analysis

* TF-IDF vectorizer built on:

  * Persona + job text
  * All section texts
* Cosine similarity computed using `scikit-learn`

### ✨ Keyword Boosting

* Keywords extracted from persona/job using regex
* Boost added to score for keyword presence in each section

### 📃 Paragraph-Level Analysis

* Top-ranked sections split into paragraphs
* Paragraphs scored semantically against persona-job
* Top 2 paragraphs extracted as refined summaries

---

## 🛠️ Tools & Technologies Used

| Category          | Tool / Library                                |
| ----------------- | --------------------------------------------- |
| 👾 Embeddings     | `sentence-transformers`, ONNX                 |
| 🔢 Vector Search  | `TF-IDF`, `cosine_similarity`, `scikit-learn` |
| 📄 PDF Parsing    | `PyMuPDF` (fitz)                              |
| 🔎 NLP Utils      | `nltk`, custom regex, text cleanup            |
| 🏗️ Model Serving | Docker, Python 3.9-slim image                 |

---

## 📅 Folder Structure

```
document-analyzer/
├── app/
│   ├── input.json
│   ├── *.pdf
│   ├── model_onnx/
│   └── output.json
├── script.py
├── requirements.txt
├── Dockerfile
├── approach_explanation.md
└── my-extractor-image.tar
```

---

## 🚪 Input Format

```json
{
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  },
  "documents": [
    { "filename": "South of France - Cities.pdf" }
  ]
}
```

---

## 🛠️ How to Run



#### 🚚 Pull Prebuilt Image from Docker Hub

```bash
docker pull yashsharma00777/pdf-extractor:latest
docker run --rm -v "$(pwd)/app:/app" yashsharma00777/pdf-extractor
```

> 📊 **Why use DockerHub image?**
>
> * Saves build time
> * Works on any Docker-enabled system
> * Preloaded model & dependencies

---

## 🔎 Sample Output (`output.json`)

```json
{
  "metadata": { ... },
  "extracted_sections": [
    {
      "document": "South of France - Cities.pdf",
      "section_title": "Day 2: Exploring Nice",
      "importance_rank": 1,
      "page_number": 3
    },
    ...
  ],
  "subsection_analysis": [
    {
      "refined_text": "The old town of Nice offers local food tours...",
      "page_number": 3
    }
  ]
}
```

---

## 🛠️ Troubleshooting

| Issue                 | Fix                                                                      |
| --------------------- | ------------------------------------------------------------------------ |
| No sections extracted | Make sure the PDF is text-based. Try: `fitz.open('file.pdf').get_text()` |
| Output is empty       | Check `input.json` structure and filenames                               |
| Model not found       | Confirm `model_onnx/` is present in `app/`                               |
| Slow processing       | Limit PDF count or shorten section length for inference                  |

---

## 🏆 Why It Stands Out

* ✅ **Smart heading detection** with spacing, font, and date filtering
* ✅ **Semantic + TF-IDF scoring hybrid**
* ✅ **Keyword-aware boosting logic**
* ✅ **Fully offline**, fast, and deployable
* ✅ **Lightweight Docker image** with only what’s necessary

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for full details.

---

> Built with ❤️ to understand documents the way humans do.
