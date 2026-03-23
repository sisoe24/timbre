# LLM-Powered Database Search & Validation — Findings & Recommendations

> Context: Audio Analyzer project — searching an audio catalog and validating CLAP-generated UCS metadata using natural language queries.

---

## Two Search Approaches

### 1. Text-to-SQL
Translates a plain-English question into a SQL query executed against the structured database.

Best for precise, filter-based queries:
- *"Find all metallic impact sounds with confidence above 0.8"*
- *"List all sounds tagged as 'reverb' sorted by confidence"*

### 2. Semantic / Vector Search
Converts descriptions into vector embeddings and matches queries by meaning rather than exact keywords.

Best for fuzzy, conceptual queries:
- *"Something that sounds industrial and harsh"*
- *"Short percussive hits with a bright tone"*

---

## Model Comparison

| Task                   | Best Cloud                      | Best Local (Free)             |
| ---------------------- | ------------------------------- | ----------------------------- |
| Text-to-SQL            | OpenAI `gpt-4o-mini`            | `defog/sqlcoder2` via Ollama  |
| Embeddings             | OpenAI `text-embedding-3-small` | `nomic-embed-text` via Ollama |
| Validation / Reasoning | Claude Sonnet / `gpt-4o`        | `llama3.1:8b` via Ollama      |

**Local vs. Cloud trade-off:**

- **Local (Ollama)** — free, private, requires a decent GPU (8GB+ VRAM), slightly lower accuracy
- **Cloud (OpenAI / Anthropic)** — best accuracy, negligible cost at this scale (~$0.00015/query), data sent externally

---

## Recommendation for This Project

A **hybrid approach** offers the best balance:

- **Structured queries** → OpenAI `gpt-4o-mini` (text-to-SQL). Near-zero cost, high accuracy.
- **Semantic search** → `nomic-embed-text` via Ollama + **ChromaDB** as the vector store. Free, local, genuinely high quality.
- **CLAP validation** → Claude Sonnet or `gpt-4o` for cloud; `llama3.1:8b` via Ollama for local.

**If full local/private setup is preferred:**
→ `defog/sqlcoder2` for SQL + `nomic-embed-text` for embeddings + `llama3.1:8b` for validation, all via Ollama.

---

## CLAP Output Validation Layer (LLM-as-Judge)

After the audio analyzer generates a UCS record, a second LLM pass reviews it for consistency and quality.

### What It Checks

- `keywords` relevance — do they actually match the `description`?
- `fx_name` accuracy — does the short title correctly represent the sound?
- `category` / `subcategory` correctness — does the UCS classification fit the description?
- Keyword redundancy — e.g. `impact`, `metallic impact`, `metal impact` all meaning the same thing
- `sound_events` vs. `description` consistency — do the temporal events match what is described?
- Confidence plausibility — is a high confidence score warranted given the description quality?

### Two Modes

**Audit mode** — LLM returns a report of issues and suggestions without modifying the original. Best for reviewing existing catalog data in bulk.

**Auto-correct mode** — LLM returns a corrected JSON record. Useful when combined with a human diff review before writing to the catalog.

### Validator Output Schema

```json
{
  "file_name": "metal_impact_01.wav",
  "issues": [
    "Keyword 'reverb' contradicts description which mentions only a brief echo",
    "'metallic impact' and 'impact' are redundant — prefer the more specific one"
  ],
  "suggested_keywords": ["impact", "metallic", "percussive", "sharp transient", "echo"],
  "suggested_category": "IMPACTS",
  "suggested_subcategory": "METAL",
  "consistency_score": 0.76,
  "notes": "Description is accurate. Keywords need deduplication and one contradiction resolved."
}
```

### Key Rules for Validation

- **Use a different model** than the one that generated the data — same model will tend to agree with itself
- **Batch carefully** — run validation as a separate async pass, not inline during generation
- **Treat auto-correct as a suggestion** — always diff original vs. corrected before writing to catalog
- **Inject the UCS spec context** into the prompt so the model understands category naming conventions

---

## Key Things to Watch Out For (All Approaches)

- Always inject the **current DB schema** into the prompt for text-to-SQL — never hardcode it
- Always **validate generated SQL** before executing (use a read-only DB user)
- Use the **same embedding model** for both indexing and querying — swapping models invalidates the index
- Ollama requires the server to be running — add a health check (`GET localhost:11434/api/tags`)
- For validation, prompt the model to return **strict JSON only** to avoid parsing issues
