import re
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    # remove html if present
    if "<html" in text.lower():
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator="\n")

    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def chunk_text(text, max_chars=1200, overlap=200):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += "\n\n" + para
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para

    if current:
        chunks.append(current.strip())

    return chunks

def infer_metadata(filename):
    name = filename.lower()

    if "missing" in name:
        topic = "missing_values"
    elif "outlier" in name:
        topic = "outliers"
    elif "correlation" in name:
        topic = "correlation"
    elif "visual" in name:
        topic = "visualization"
    elif "workflow" in name:
        topic = "workflow"
    else:
        topic = "eda_general"

    return {
        "doc_type": "eda_guideline",
        "topic": topic,
        "source": "eda_handbook"
    }