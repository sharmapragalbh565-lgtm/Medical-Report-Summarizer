# Medical Report Summarizer

(Powered by FLAN-T5 Large)

Project Overview:

This project automatically extracts and summarizes medical reports (PDF or text files) using a Large Language Model from Google — specifically FLAN-T5 Large.

It is designed to:

1. Extract text from medical PDFs

2. Generate concise medical summaries

3. Reduce manual reading time

4. Assist doctors, healthcare analysts, and medical researchers

# Features

Extracts text from PDF medical reports

Handles large documents via chunking

Generates structured summaries

Works in Google Colab

Supports both PDF and TXT files

# Tech Stack

Python

Hugging Face Transformers

PyTorch

PyPDF

SentencePiece

# Installation
pip install transformers accelerate sentencepiece pypdf

# How It Works
1️. Load Model
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

2️. Extract Text from PDF
reader = PdfReader(pdf_path)

Extracts text page-by-page.

3️. Handle Long Reports (Chunking)
chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]

Large medical documents are split into manageable chunks to avoid model token limits.

4️. Generate Summary
summarizer(chunk, max_length=150, min_length=50)

Each chunk is summarized and combined into a final summary.

# Project Structure
medical_report_summarizer/
│
├── summarizer.py
├── requirements.txt
└── README.md

# Example Workflow

Upload a medical PDF in Colab

Text is extracted

Model generates structured summary

Output printed in notebook

# Use Cases

Hospital documentation automation

Medical research assistance

Clinical decision support

Patient discharge summary generation

Insurance claim review

Medical record compression


# Possible Improvements

Fine-tune model on medical datasets

Add named entity recognition (diseases, medications)

Export summary to PDF

Build Streamlit web interface

Add multilingual support

Deploy as REST API

Add risk classification layer


# Future Scope

Integration with hospital EMR systems

Real-time clinical report summarization

Voice-based report reading

AI-powered medical insights dashboard
