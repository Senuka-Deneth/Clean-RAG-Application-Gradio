import gradio as gr
import faiss
import numpy as np
import requests
from pypdf import PdfReader

# Create a memory store for retrieved chunks and the vector index
store = {
    "chunks" : [],
    "index" : None,
    "dim" : None,
}

#Read text from the uploaded PDF
def read_pdf(path : str) -> str:
    reader = PdfReader(path,strict=False)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

#Split the document text into chunks
def chunk_text(text : str, chunk_size=800, overlap=150):
    chunks = []
    i = 0
    while (i<len(text)):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        i+= chunk_size - overlap
    return chunks

#Embed the chunks (convert text to numerical vectors)
def embed_texts(texts, model="nomic-embed-text:latest"):
    url = "http://localhost:11434/api/embeddings"
    vecs = []
    for t in texts:
        r = requests.post(url, json={"model" : model.strip(), "prompt" : t}, timeout=120)
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
    return np.array(vecs, dtype=np.float32)

#Build a FAISS index from the chunk embeddings
def build_index(chunks):
    vecs = embed_texts(chunks)
    dims = vecs.shape[1]

    index = faiss.IndexFlatIP(dims)
    faiss.normalize_L2(vecs)
    index.add(vecs)

    store["chunks"]=chunks
    store["dim"]=dims
    store["index"]=index

    return f"Loaded {len(chunks)} chunks."

#Ingestion done (load → chunk → embed → index)
#Query side (retrieve → answer)

#Retrieve the most relevant chunks for the user query
def retrieve(query, top_k=5):
    if store["index"] is None:
        return []
    
    qvec = embed_texts([query])
    faiss.normalize_L2(qvec)

    scores, ids = store["index"].search(qvec, top_k)
    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append((float(score), store["chunks"][idx]))
    return results

#Sends the prompt to ollama
def ollama_chat(prompt : str, model : str = "gpt-oss:20b") -> str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model" : model.strip(),
        "messages" : [{"role" : "user", "content":prompt}],
        "stream" : False,
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

#Format short citations (small quotes) for the answer
def format_citations(results):
    out = []
    for i, (score, txt) in enumerate(results[:5], start=1):
        quote = " ".join(txt.strip().split())
        quote = quote[:220] + ("..." if len(quote) > 220 else "")
        out.append(f"[{i}] score={score:.3f} | \"{quote}\"")
    return "\n".join(out)

#Full RAG pipeline (retrieve → check strength → answer → citations)
def answer_question(query, top_k=5, llm_model="gpt-oss:20b"):
    results = retrieve(query, top_k=top_k)

    if not results:
        return "I don't know (no documents loaded).", ""

    best_score = results[0][0]
    if best_score < 0.2:
        return "I don't know (retrieval seems weak).", format_citations(results)

    context = "\n\n".join([f"[{i+1}] {txt}" for i, (_, txt) in enumerate(results)])

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
"""
    ans = ollama_chat(prompt, model=llm_model)
    return ans, format_citations(results)

#Ingest the uploaded file into the knowledge base
def ingest_file(file):
    if file is None:
        return "No file."

    path = file.name
    if path.lower().endswith(".pdf"):
        text = read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    chunks = chunk_text(text)
    return build_index(chunks)

#Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Clean RAG (Gradio + Ollama local LLM)")

    with gr.Row():
        file_in = gr.File(label="Upload PDF or TXT")
        load_btn = gr.Button("Ingest")

    status = gr.Textbox(label="Status")
    gr.Markdown("## Ask")
    question = gr.Textbox(label="Question")
    top_k = gr.Slider(1, 10, value=5, step=1, label="Top-k")
    llm_model = gr.Textbox(label="Ollama LLM model name", value="gpt-oss:20b")
    ask_btn = gr.Button("Answer")

    answer = gr.Textbox(label="Answer", lines=6)
    citations = gr.Textbox(label="Citations (quotes)", lines=8)

    load_btn.click(ingest_file, inputs=[file_in], outputs=[status])
    ask_btn.click(answer_question, inputs=[question, top_k, llm_model], outputs=[answer, citations])

demo.launch()
