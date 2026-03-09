# ===== standard libs =====
import os
import re
import json
import pickle
from typing import Literal

import numpy as np

# ===== Vertex =====
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput

# ===== utils =====
from pydantic import BaseModel

PROJECT_ID = PROJECT_ID
LOCATION = LOCATION

DEBUG = False

LOCAL_INDEX_PATH = "data/financial_reports_faiss.pkl"

vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
llm = GenerativeModel("gemini-2.5-pro")

def load_local_index():
    with open(LOCAL_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    index = data["index"]
    chunk_ids = data["chunk_ids"]
    metadata = data["metadata"]
    titles = data["titles"]
    contents = data["contents"]

    return index, chunk_ids, metadata, titles, contents

_index_cache = None

def get_faiss_index():
    global _index_cache
    if _index_cache is None:
        _index_cache = load_local_index()
    return _index_cache

# Construct a compact, LLM-friendly context from retrieved document chunks
def build_context(retrieved_chunks, max_chunks=12, max_chars_per_chunk=1500):
    context_blocks = []

    for r in retrieved_chunks[:max_chunks]:
        content = r["content"]

        if len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk] + "..."

        block = f"""
[Document]
Company: {r['metadata'].get('company', 'N/A')}
Year: {r['metadata'].get('year', 'N/A')}
Section: {r['metadata'].get('item', '')}

Content:
{content}
"""
        context_blocks.append(block.strip())

    return "\n\n".join(context_blocks)

# Normalize company names to improve matching across filings and queries
def normalize_company(name: str) -> str:
    if not name:
        return ""

    name = name.lower()

    suffixes = [
        "corporation",
        "corp",
        "incorporated",
        "inc",
        "ltd",
        "limited",
        "company",
        "co",
        "plc",
        "holdings",
        "group"
    ]

    name = re.sub(r"[.,]", "", name)

    for s in suffixes:
        if name.endswith(" " + s):
            name = name[: -len(s) - 1]

    name = re.sub(r"\s+", " ", name).strip()

    return name

# Retrieval strategy for questions focused on a single company
# Emphasizes high recall followed by hard metadata filtering
def retrieve_single_company(question, plan, top_k=100):
    # Step 1: recall-oriented retrieval
    results = retrieve_top_k_chunks(
        plan.retrieval_query or question,
        top_k=500
    )

    # Step 2: HARD filter by company
    if plan.companies:
        normalized_plan_companies = {
            normalize_company(c) for c in plan.companies
        }

        results = [
            r for r in results
            if normalize_company(r["metadata"].get("company", ""))
               in normalized_plan_companies
        ]

    # Step 3: HARD filter by item (e.g., Item 1A)
    if plan.item:
        results = [
            r for r in results
            if r["metadata"].get("item", "").startswith(plan.item)
        ]

    return results[:top_k]

# Retrieval strategy for multi-company comparison questions
# Ensures balanced coverage by limiting chunks per company
def retrieve_multi_company(
    question: str,
    plan,
    k_per_company: int = 6,
    candidate_pool: int = 500
):
    retrieval_query = plan.retrieval_query or question

    candidates = retrieve_top_k_chunks(
        retrieval_query,
        top_k=candidate_pool
    )

    # 3. Per-company slicing (coverage constraint)
    results = []
    for company in plan.companies:
        normalized_company = normalize_company(company)
        company_chunks = [
            r for r in candidates
            if normalize_company(r["metadata"].get("company", "")) == normalized_company
        ][:k_per_company]

        results.extend(company_chunks)

    return results

# General retrieval strategy when no specific company constraints apply
def retrieve_general(question, plan, top_k=80):
    retrieval_query = plan.retrieval_query if plan.retrieval_query else question
    return retrieve_top_k_chunks(
        retrieval_query,
        top_k=top_k
    )

def embed_query(query: str) -> np.ndarray:
    query_input = TextEmbeddingInput(
        task_type="RETRIEVAL_QUERY",
        text=query
    )
    embedding = embedding_model.get_embeddings([query_input])[0].values
    return np.array(embedding, dtype="float32")

def split_into_sentences(text: str):
    return re.split(r"(?<=[.!?])\s+", text)

# Extract sentence-level evidence by re-ranking sentences within a chunk
# using semantic similarity to the user question
def extract_sentence_evidence(
    chunk_text: str,
    question: str,
    top_n: int = 2,
    max_sentences: int = 120
):
    sentences = split_into_sentences(chunk_text)

    if len(sentences) > max_sentences:
        head = sentences[: max_sentences // 2]
        tail = sentences[- max_sentences // 2 :]
        sentences = head + tail

    if not sentences:
        return []

    q_emb = embed_query(question)
    sent_embs = embedding_model.get_embeddings(sentences)

    scored = []
    for s, emb in zip(sentences, sent_embs):
        vec = np.array(emb.values, dtype="float32")
        score = float(np.dot(vec, q_emb))
        scored.append((s, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]]

# Low-level vector search over the FAISS index to retrieve relevant chunks
def retrieve_top_k_chunks(query: str, top_k: int = 8):
    query_vector = embed_query(query).reshape(1, -1)
    query_vector = np.array(query_vector, dtype="float32")
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    index, chunk_ids, metadata, titles, contents = get_faiss_index()
    top_k = min(top_k, len(chunk_ids))
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        results.append({
            "rank": rank + 1,
            "chunk_id": chunk_ids[idx],
            "title": titles[idx],
            "metadata": metadata[idx],
            "content": contents[idx],
            "distance": float(distances[0][rank]),
        })

    return results

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    return json.loads(match.group())

# Structured query plan produced by the LLM-based planner
class QueryPlan(BaseModel):
    retrieval_query: str
    companies: list[str] | None = None
    start_year: int | None = None
    end_year: int | None = None
    item: str | None = None
    intent: Literal["single_company", "multi_company", "general"]


# Use an LLM to parse the user question into a structured retrieval plan
def plan_query(question: str, llm) -> QueryPlan:
    prompt = QUERY_PLANNER_PROMPT.format(question=question)

    response = llm.generate_content(prompt)
    text = response.text

    plan_dict = extract_json(text)
    return QueryPlan(**plan_dict)

# Fallback mechanism to relax overly strict constraints when retrieval fails
def relax_query_plan(plan: QueryPlan, original_question: str) -> tuple[QueryPlan, str]:
    """
    Relax overly restrictive constraints in the query plan.
    Returns (new_plan, rewritten_question)
    """
    new_plan = plan.model_copy(deep=True)

    new_plan.retrieval_query = original_question

    new_plan.item = None

    new_plan.start_year = None
    new_plan.end_year = None

    return new_plan, original_question

# Lightweight heuristic to infer the analysis style for answer generation
def infer_analysis_type(question: str) -> str:
    q = question.lower()

    if any(k in q for k in [
        "drive", "drove", "growth", "increase", "decline",
        "revenue", "profit", "margin"
    ]):
        return "business_driver"

    if any(k in q for k in [
        "what is", "define", "description", "role", "function"
    ]):
        return "definition"

    if "risk" in q or "uncertainty" in q:
        return "risk_factor"

    if "compare" in q or "difference" in q:
        return "comparison"

    return "definition"

QUERY_PLANNER_PROMPT = """
You are a query planner for a RAG system.

First classify the user question intent as one of:
- "single_company": focused on one company
- "multi_company": explicitly comparing or asking about multiple companies
- "general": industry-wide or no specific company

If multiple companies are mentioned, extract ALL of them.

Then extract from the user question:
- A short retrieval query optimized for semantic search
- The company name if mentioned
- The relevant year range if mentioned
- The report section if mentioned (e.g., Item 1, Item 7)

You MUST return a JSON object with EXACTLY the following fields:
{{
  "retrieval_query": string,
  "companies": array of strings or null,
  "start_year": number or null,
  "end_year": number or null,
  "item": string or null,
  "intent": "single_company" | "multi_company" | "general"
}}

Rules:
- retrieval_query is REQUIRED and must NOT be empty.
- Use "item" for report sections (e.g., "Item 1A", "Item 7").
- Do NOT invent new field names (e.g., do NOT use "section").

User Question:
{question}
"""

# Build the final prompt for the generative model, including evidence rules
# and analysis-specific instructions
def build_rag_prompt(context: str, question: str, analysis_type: str) -> str:

    if analysis_type == "business_driver":
        instructions = """
If the question asks about business performance, growth, or major trends:

- First, IDENTIFY AND ENUMERATE the MAJOR TRENDS explicitly highlighted
  across the documents. Multiple trends may coexist.
- Do NOT collapse multiple recurring themes into a single point unless
  the documents clearly emphasize only one dominant trend.
- Then, SYNTHESIZE these trends into a higher-level analytical summary
  that explains the company’s strategic direction or business trajectory.
- Focus on economic or strategic patterns (e.g., demand shifts, business
  model transitions, recurring revenue, customer adoption), not wording.
- The answer should resemble an analyst’s thematic summary rather than
  a single-sentence conclusion.
- All analysis MUST be grounded in the provided evidence, but you are
  encouraged to interpret and generalize at a higher level.
"""

    elif analysis_type == "definition":
        instructions = """
If the question asks what a technology, platform, or product is:

- Describe its FUNCTIONAL ROLE, PURPOSE, and CAPABILITIES
  as defined in the documents.
- Focus on what it enables and how the company positions it.
- Do NOT focus on financial performance unless explicitly asked.
"""

    elif analysis_type == "risk_factor":
        instructions = """
If the question is about risks or uncertainties:

- Summarize the key risks described in the documents.
- Use the company's own framing and language.
"""

    else:
        instructions = """
Answer the question using the provided context.
"""

    # Insert the GLOBAL EVIDENCE PRIORITY RULE after {instructions}, before Note:
    global_evidence_priority = """
GLOBAL EVIDENCE PRIORITY RULE (applies to ALL questions):

- If the context contains one or more sentences that explicitly and directly
  answer the user’s question, those sentences MUST form the core basis
  of the answer.
- Do NOT ignore or override such sentences in favor of more abstract,
  generalized, or indirect descriptions.
- You may paraphrase or summarize these sentences at a higher level,
  but the main conclusion of the answer must clearly reflect them.
- Only if NO such explicit sentences exist should you rely on
  indirect inference or broader contextual analysis.
"""

    prompt = f"""
You are a financial and technology analyst.

The documents are 10-K annual reports of major technology companies.
Although these documents are financial in nature, they often contain
IMPORTANT descriptions of technology platforms, products, and services.

{instructions}
{global_evidence_priority}
Note:
- Annual reports are published the year AFTER the fiscal year discussed.
- For example, fiscal year 2020 results appear in the 2021 annual report.

Use ONLY the information provided in the context below.
Do NOT use outside knowledge.
Only state that "the documents do not provide sufficient information"
IF AND ONLY IF the context does not contain any sentence that directly
or implicitly answers the question.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

def generate_answer(prompt: str) -> str:
    response = llm.generate_content(prompt)
    return response.text

RETRIEVAL_STRATEGIES = {
    "single_company": retrieve_single_company,
    "multi_company": retrieve_multi_company,
    "general": retrieve_general,
}

# End-to-end RAG pipeline: plan → retrieve → extract evidence → generate answer
def answer_user_question(question: str):
    # ===== First attempt =====
    plan = plan_query(question, llm)

    # Override planner query with the original question to preserve sentence-level recall
    # ===== Base retrieval using original question (critical for MD&A sentences) =====
    plan.retrieval_query = question
    base_results = RETRIEVAL_STRATEGIES[plan.intent](question, plan)
    all_results = list(base_results)

    seen = set()
    retrieved_chunks = []
    for r in all_results:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            retrieved_chunks.append(r)

    # ===== Default bias: if no explicit company & year, search everything =====
    # If neither company nor year is specified, use the general retrieval strategy
    no_company_specified = not plan.companies
    no_year_specified = plan.start_year is None and plan.end_year is None

    if no_company_specified and no_year_specified:
        plan.intent = "general"

    # ===== Fallback: relax overly restrictive constraints =====
    # Apply fallback retrieval if all constraints eliminate relevant evidence
    if not retrieved_chunks:
        print("No meaningful retrieval results. Applying general fallback.")

        fallback_plan, fallback_question = relax_query_plan(plan, question)

        retrieved_chunks = retrieve_general(
            fallback_question,
            fallback_plan
        )

        question_to_answer = fallback_question
    else:
        question_to_answer = question

    # ===== Generate answer =====
    # Determine the analysis style for answer generation
    analysis_type = infer_analysis_type(question_to_answer)

    if analysis_type == "business_driver":
        # Prioritize MD&A causal sentences (e.g., usage, pricing)
        retrieved_chunks = sorted(
            retrieved_chunks,
            key=lambda r: r["distance"]
        )
    else:
        retrieved_chunks = sorted(
            retrieved_chunks,
            key=lambda r: (
                r["metadata"].get("company", ""),
                r["metadata"].get("year", 0)
            )
        )

    if DEBUG:
        for r in retrieved_chunks[:20]:
            print("-----")
            print(r["content"])
            print(r["metadata"].get("company"),
                  r["metadata"].get("item"),
                  r["distance"])

    evidence_blocks = []

    for r in retrieved_chunks:
        chunk_text = r["content"]
        evidence = extract_sentence_evidence(
            chunk_text,
            question_to_answer,
            top_n=4 if analysis_type in ["business_driver", "definition"] else 2
        )

        if evidence:
            evidence_blocks.append(
                f"""
    [EXPLICIT EVIDENCE]
    Company: {r['metadata'].get('company')}
    Year: {r['metadata'].get('year')}
    Section: {r['metadata'].get('item')}

    {" ".join(evidence)}
    """.strip()
            )

    chunk_context = build_context(
        retrieved_chunks,
        max_chunks=20,
        max_chars_per_chunk=5000
    )

    context = "\n\n".join(evidence_blocks) + "\n\n" + chunk_context

    prompt = build_rag_prompt(
        context,
        question_to_answer,
        analysis_type
    )

    return generate_answer(prompt)

if __name__ == "__main__":
    print(answer_user_question("What factors drove AWS revenue growth?"))
