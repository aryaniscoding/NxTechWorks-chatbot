# llm_utils.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Configure with your API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def llm_answer(question: str, context_chunks: list[str]) -> str:
    # Configure the API key (make sure it's set before calling this function)
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create the model instance
    model = genai.GenerativeModel("gemma-3-12b-it")
    
    # Build the prompt
    prompt = "Use the context to answer. Context:\n"
    for chunk in context_chunks:
        prompt += f"- {chunk}\n"
    prompt += f"\nQuestion: {question}\nAnswer:"
    
    # Generate content with temperature setting
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2
        )
    )
    
    return response.text

def contextual_answer(question: str, context_chunks: list[str], history: list[tuple[str,str]]) -> str:
    """
    Build a prompt with:
      1. System instruction
      2. The last few (user, assistant) pairs
      3. Retrieved context chunks
      4. The new user question
    """
    model = genai.GenerativeModel("gemma-3-12b-it")
    # 1) System
    prompt = "You are a helpful assistant in a document-Q&A session.\n"
    # 2) History
    if history:
        prompt += "Here is the recent conversation:\n"
        for q, a in history:
            prompt += f"User: {q}\nAssistant: {a}\n"
    # 3) Context
    prompt += "\nUse the following context to answer the next question:\n"
    for c in context_chunks:
        prompt += f"- {c}\n"
    # 4) New question
    prompt += f"\nUser: {question}\nAssistant:"
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.2)
    )
    return resp.text.strip()


def generate_questions(full_text: str) -> list[str]:
    """
    Generates three relevant questions a user might ask based on the document text.
    """
    model = genai.GenerativeModel("gemma-3-12b-it")
    # Truncate to first 2000 characters to avoid overly long prompt
    context = full_text[:]
    prompt = (
        "Based on the following document content, suggest three insightful frequently asked questions that a user could ask the chatbot about this document. "
        "Present each frequently question as a short sentence.\n\n"
        f"Document excerpt:\n{context}\n\nQuestions:"
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.2)
    )
    # Split lines and take non-empty first three
    questions = [q.strip('- ').strip() for q in response.text.splitlines() if q.strip()]
    return questions[1:4]


def generate_summary(chunks: list[str]) -> str:
    # Map: summarize each chunk
    mini_summaries = []
    for chunk in chunks:
        prompt = (
            "Write a concise 2–3 sentence summary of the following text:\n\n"
            f"{chunk}\n\nSummary:"
        )
        resp = llm_answer(prompt, [])  # or a dedicated summarization call
        mini_summaries.append(resp.strip())

    # Reduce: combine and summarize again
    combined = "\n".join(mini_summaries)
    final_prompt = (
        "Here are summaries of sections from a larger document. "
        "Please write a single coherent paragraph capturing the overall main points:\n\n"
        f"{combined}\n\nOverall Summary:"
    )
    return llm_answer(final_prompt, [])


