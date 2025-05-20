# file_utils.py

import fitz  # PyMuPDF
import docx
import pandas as pd
import sqlite3
from io import BytesIO

def detect_file_type(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return "pdf"
    elif name.endswith(".docx"):
        return "docx"
    elif name.endswith(".csv"):
        return "csv"
    elif name.endswith(".xls") or name.endswith(".xlsx"):
        return "excel"
    elif name.endswith(".db"):
        return "sqlite"
    return "unknown"

def get_page_count(file_stream, ftype):
    if ftype == "pdf":
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        return doc.page_count
    elif ftype == "docx":
        return 1  # DOCX: no page concept, treat as 1
    elif ftype in ["csv", "excel", "sqlite"]:
        return 1
    return 0

def get_text(file_stream, ftype):
    file_stream.seek(0)
    if ftype == "pdf":
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif ftype == "docx":
        doc = docx.Document(file_stream)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ftype == "csv":
        df = pd.read_csv(file_stream)
        return df.to_string(index=False)
    elif ftype == "excel":
        df = pd.read_excel(file_stream)
        return df.to_string(index=False)
    elif ftype == "sqlite":
        conn = sqlite3.connect(':memory:')
        with open(file_stream.name, 'rb') as f:
            data = f.read()
        with open('temp.db', 'wb') as temp:
            temp.write(data)
        conn = sqlite3.connect('temp.db')
        text = ""
        cursor = conn.cursor()
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"):
            table = row[0]
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 100", conn)
            text += f"\nTable: {table}\n" + df.to_string(index=False)
        conn.close()
        return text
    return ""
