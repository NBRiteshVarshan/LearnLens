# LearnLens — Exam Intelligence Platform

Me and my friend built this during the Sprint4Good AI Hackathon at IIT Delhi — a 12 hour hackathon where we had to build something from scratch and present it the same day. We were a team of just 2 first-years, surrounded by bigger, more experienced teams from all corners of the country. Tough crowd. But we gave it our best shot.

The idea was simple — students waste a lot of time searching Google or asking ChatGPT questions that aren't even aligned to their syllabus. So we built LearnLens, an AI platform where you upload your own notes and the platform answers only from those notes. No random internet stuff. Just your syllabus, your questions, your prep.

---

## What LearnLens can do

### Upload your Notes
You upload a PDF of your notes, hit "Ingest PDF", and the platform breaks it into chunks, embeds them, and stores everything in MongoDB. That's it. Your notes are now searchable.

### Ask Questions
This is the main feature. You type a question and the platform searches through your notes using vector search and gives you an answer — with the page number where it found it. If the answer isn't in your notes, it will straight up tell you that instead of making something up. Oh, and you can also ask in Hinglish. We added that because, well, why not.

### Summarise
Gives you a clean, structured summary of your uploaded notes — key concepts, definitions, formulas, and a few quick revision points. Useful when you just want a quick refresher before an exam.

### Quiz
Probably the coolest feature we added. You can generate 10 MCQs from your notes and attempt them with a live score. There are three difficulty levels — Easy, Medium, and Hard. And if you have your previous year question papers, you can upload those too and the platform will generate MCQs that follow the same pattern as your PYQs. That one turned out really well.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Ollama — gemma3:4b |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 |
| Database | MongoDB Atlas (Vector Search) |
| PDF Parsing | PyMuPDF |
| Language | Python |

It was our first time using MongoDB and honestly it took some time to get going. But once we figured out the MONGO_URI and got the vector index set up, everything started clicking.

---

## Setup

### What you need before starting
- Python 3.9+
- Ollama installed and running on your machine
- A MongoDB Atlas account with Vector Search enabled

### Steps

**1. Clone the repo**
```bash
git clone https://github.com/your-username/learnlens.git
cd learnlens
```

**2. Install the dependencies**
```bash
pip install streamlit pymongo sentence-transformers ollama pymupdf
```

**3. Pull the model**
```bash
ollama pull gemma3:4b
```

**4. Add your MongoDB URI**

In `app.py`, find this line and replace it with your actual connection string:
```python
MONGO_URI = "your-mongodb-connection-string-here"
```

**5. Create the Vector Search Index**

Go to your MongoDB Atlas dashboard, open the `chunks` collection, and create a Vector Search Index with this config:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

Name it `vector_index`. This step is easy to miss but without it, the search won't work.

**6. Run the app**
```bash
streamlit run app.py
```

---

## How it works under the hood

When you ingest a PDF, each page is parsed and split into overlapping chunks of around 220 words (with a 40 word overlap so context doesn't get cut off at boundaries). Each chunk gets embedded using `all-MiniLM-L6-v2` and stored in MongoDB.

When you ask a question, the same embedding model converts your question into a vector and MongoDB's `$vectorSearch` finds the most relevant chunks from your notes. Those chunks are passed to the Ollama LLM as context, which then generates an answer — strictly based on what's in your notes.

The quiz generator asks the LLM to return exactly 10 MCQs in JSON format. When you upload a PYQ, the LLM analyses its pattern and generates new questions from your notes that follow the same style.

---

## Project Structure

```
learnlens/
│
├── app.py          # Everything lives here
├── uploads/        # PDFs get saved here temporarily (auto-created)
└── README.md
```

---

## What we would improve if we had more time

- Proper user login so multiple people can use it without data mixing
- Support for multiple PDFs in a single session
- A proper frontend — Streamlit worked fine for the hackathon but we would eventually want something better
- Flashcard mode
- Deployment so anyone can use it without setting it up locally

---

## Hackathon Context

This was built at the Sprint4Good AI Hackathon organised by IIT Delhi. We had 12 hours, a hall full of participants who had clearly been building their projects for months, and a team of just 2. For starters, every other team had at least 3 members. Also we were both first-years, so the lack of experience in competitive hackathons was definitely a factor.

We did not make it to the top 6. But a mentor did compare our Ask Question feature with ChatGPT live, and our response turned out to be more syllabus-aligned. That felt good.

We came back with a lot of ideas and honestly, more drive than we left with.
