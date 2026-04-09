!pip install -q transformers accelerate sentencepiece pypdf


from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pypdf import PdfReader


model_name = "google/flan-t5-large"  # Lightweight; good performance
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def summarize_medical_report(text):
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return "\n".join(summaries)


from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith(".pdf"):
        print(f"\nExtracting text from {filename}...")
        text_data = extract_text_from_pdf(filename)
    else:
        print(f"\nReading text from {filename}...")
        with open(filename, 'r') as f:
            text_data = f.read()


print("\nGenerating summary...")
summary = summarize_medical_report(text_data)
print("\nSummary:\n")
print(summary)
