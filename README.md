
# Document Section Extractor and Relevance Scorer

This project extracts sections from PDF documents and ranks them for relevance to a specified job role (e.g., _"Travel Planner planning a 4-day trip for 10 college friends"_) using semantic embeddings and TF-IDF scoring.

✅ **Offline execution** | 💻 **CPU-only** | 🧠 **Model size ≤ 1GB** | ⚡ **<60s for 3–5 PDFs**

---

## 🚀 Features

- **PDF Section Extraction**: Detects headings using:
  - Bold/italic font styles
  - Vertical spacing (>30 units)
  - 2–15 word titles (excluding dates)
- **Relevance Scoring**:
  - Semantic embeddings via `all-mpnet-base-v2` (ONNX, 0.85 weight)
  - TF-IDF scoring (0.15 weight)
  - TF-IDF-weighted keyword boosting for job alignment
- **Clean Output**:
  - JSON output (`output.json`) with ranked sections & relevant paragraphs
  - No extra formatting (e.g., bullets, newlines)

---

## 📦 Prerequisites

- **Docker** (for containerized execution)
- **Python** 3.9+ (handled inside Docker)
- **Input Files**:
  - `input.json`: Includes persona, job-to-be-done, and list of PDFs
  - PDFs: Must be **text-based**, not scanned images

### ✅ Example `input.json`

json
{
  "persona": { "role": "Travel Planner" },
  "job_to_be_done": { "task": "Plan a trip of 4 days for a group of 10 college friends." },
  "documents": [
    { "filename": "South of France - Cities.pdf" }
  ]
}
`

---

## ⚙ Setup Instructions

### 1. Clone the Repository

bash
git clone https://github.com/your-username/document-section-extractor.git
cd document-section-extractor


### 2. Prepare Input Files

* Place `input.json` and all PDF files into the `app/` directory.

### 3. Build Docker Image

bash
docker build -t document-analyzer .


### 4. Run the Container

bash
docker run --rm -v $(pwd)/app:/app document-analyzer


* Output will be in: `app/output.json`
* No internet required after setup

---

## 🔄 Model Conversion (One-Time)

Convert `all-mpnet-base-v2` to ONNX using `optimum`:

bash
pip install optimum
python script.py


* Copy the generated `model_onnx/` folder into `app/`

---

## 📁 Project Structure


.
├── script.py                # Main script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Build instructions
├── approach_explanation.md  # Technical details
├── app/
│   ├── input.json
│   ├── *.pdf                # Input PDFs
│   ├── model_onnx/          # Converted ONNX model
│   └── output.json          # Result file


---

## 🐳 Dockerfile

Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt

COPY script.py .
COPY input.json .
COPY model_onnx/ /app/model_onnx/

CMD ["python", "script.py"]


---

## 📦 requirements.txt

txt
PyMuPDF==1.23.26
sentence-transformers==2.7.0
nltk==3.8.1
onnxruntime==1.17.1
numpy==1.26.4
scikit-learn==1.4.1.post1
optimum==1.17.1
```

---

## 🧠 Methodology

See approach_explanation.md for details on:

* Heading detection via font & spacing
* Semantic scoring using ONNX
* Keyword boosting based on job relevance
* Optimization techniques for speed and memory

---

## 🛠 Troubleshooting

| Issue                     | Solution                                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| *No sections extracted* | Verify PDF is text-based using:<br>python -c "import fitz; print(fitz.open('file.pdf').get_text())" |
| *Model errors*          | Ensure model_onnx/ exists in app/                                                                 |
| *Slow processing*       | Reduce max_length in encoding; Limit extracted sections                                             |

---

## ⚙ Constraints

* ✅ CPU-only (ONNX for fast inference)
* ✅ Model size < 1GB (~420MB)
* ✅ Processing time < 60s for 3–5 documents
* ✅ Fully offline after setup

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
