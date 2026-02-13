# HCPCS Inference Service v1

A modular Python pipeline that infers HCPCS/CPT medical procedure codes from healthcare policy text with confidence scores, justifications, and full audit trails.

## Quick Start

```bash
pip install -r requirements.txt

# Default method (TF-IDF):
python run_pipeline.py --input policies_cleaned.csv --output inferred_codes.json

# With evaluation against ground truth:
python run_pipeline.py --input policies_cleaned.csv --output inferred_codes.json --labels policies_cleaned_labels.csv

# Mock LLM (no API key needed):
python run_pipeline.py --input policies_cleaned.csv --output inferred_codes.json --method llm_mock

# Real LLM (requires OPENAI_API_KEY):
export OPENAI_API_KEY=sk-...
python run_pipeline.py --input policies_cleaned.csv --output inferred_codes.json --method llm_openai --limit 5
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to policies CSV |
| `--output` | `inferred_codes.json` | Output JSON path |
| `--hcpcs-catalog` | `hcpcs.csv` | HCPCS reference CSV |
| `--method` | `lexical_tfidf` | `lexical_tfidf`, `llm_openai`, or `llm_mock` |
| `--limit` | all | Process only first N policies |
| `--top-k` | 10 | Max codes per policy |
| `--threshold` | 0.05 | Minimum confidence threshold |
| `--labels` | *(none)* | Labels CSV for evaluation |

## Architecture

```
run_pipeline.py                     # CLI entrypoint
src/policybot_hcpcs/
  pipeline.py                       # Orchestration: load → infer → aggregate → write
  config.py                         # Constants and defaults
  models/schemas.py                 # Pydantic v2 data models (API contract)
  io/loaders.py                     # CSV loading and normalization
  io/writers.py                     # JSON output serialization
  inference/base.py                 # Abstract inference method (strategy pattern)
  inference/registry.py             # Method registration and factory
  inference/lexical.py              # TF-IDF + literal match (primary v1)
  inference/llm_openai.py           # OpenAI GPT inference
  inference/llm_mock.py             # Regex fallback (no API key)
  evaluation/metrics.py             # Recall@k, precision@k, coverage
```

### Pipeline Flow

```
1. LOAD     → Read policies CSV and HCPCS catalog into Pydantic models
2. INIT     → Instantiate inference method, perform one-time setup (e.g., fit TF-IDF)
3. INFER    → For each policy, produce CodeInference objects with provenance
4. AGGREGATE → Wrap results in InferenceOutput with run metadata
5. WRITE    → Serialize to inferred_codes.json
6. EVALUATE → (optional) Compare against ground truth labels
```

### Strategy Pattern for Inference Methods

All methods implement `BaseInferenceMethod`:

```python
class BaseInferenceMethod(ABC):
    method_type: InferenceMethodType     # Unique identifier
    def initialize(hcpcs_catalog): ...   # One-time setup
    def infer(policy, catalog, params):  # Per-policy inference → list[CodeInference]
```

New methods are registered with a decorator:

```python
@register_method(InferenceMethodType.LEXICAL_TFIDF)
class LexicalTfidfMethod(BaseInferenceMethod): ...
```

## API Contract (Output Schema)

### `inferred_codes.json` Structure

```json
{
  "run_metadata": {
    "pipeline_version": "1.0.0",
    "schema_version": "1.0.0",
    "run_timestamp": "2026-02-13T21:34:46Z",
    "method": "lexical_tfidf",
    "input_file": "policies_cleaned.csv",
    "hcpcs_catalog_file": "hcpcs.csv",
    "total_policies": 200,
    "total_inferences": 1941,
    "parameters": {"top_k": 10, "threshold": 0.05}
  },
  "results": [
    {
      "policy_id": "fffed90d-...",
      "policy_name": "Fremanezumab-vfrm (Ajovy)",
      "inferences": [
        {
          "code": "J3031",
          "code_system": "HCPCS",
          "code_description": "Injection, fremanezumab-vfrm, 1 mg",
          "confidence": 0.85,
          "justification": "HCPCS J3031 matched via TF-IDF similarity (score=0.55). Code 'J3031' found literally in policy text.",
          "provenance": {
            "pipeline_version": "1.0.0",
            "method": "lexical_tfidf",
            "timestamp": "2026-02-13T21:34:46Z",
            "inputs_hash": "f8fc8ac0dd19...",
            "parameters": {"top_k": 10, "threshold": 0.05},
            "evidence": [
              {
                "evidence_type": "tfidf_similarity",
                "source": "hcpcs_catalog",
                "snippet": "Injection, fremanezumab-vfrm, 1 mg",
                "score": 0.55
              },
              {
                "evidence_type": "text_match",
                "source": "policy_text",
                "snippet": "...HCPCS Code J3031 for injection...",
                "span": {"start": 4521, "end": 4526, "text": "J3031"}
              }
            ],
            "artifacts": {}
          }
        }
      ]
    }
  ]
}
```

### Key Schema Properties

- **`confidence`** (0.0–1.0): Base score from the inference method (TF-IDF cosine similarity or LLM self-reported), boosted +0.3 for literal code matches in text.
- **`justification`**: Human-readable explanation for each inferred code.
- **`provenance`**: Full audit object containing:
  - `method`: Which algorithm produced this result
  - `inputs_hash`: SHA-256 of (policy_text + catalog_version) for reproducibility
  - `parameters`: Exact parameters used (top_k, threshold, model, temperature)
  - `evidence`: List of supporting evidence items with types, sources, and scores
  - `artifacts`: Additional data (LLM model version, prompt hash, token usage)

## Inference Methods

### `lexical_tfidf` (default, primary v1)

1. Fits a TF-IDF vectorizer on all 1,320 HCPCS code descriptions (bigrams, 5000 features)
2. Transforms each policy text and computes cosine similarity against all descriptions
3. Takes top-k results above the confidence threshold
4. Scans for literal code patterns (e.g., `J3031`, `99213`) and boosts confidence +0.3
5. Deterministic — same input always produces same output

### `llm_openai` (requires `OPENAI_API_KEY`)

1. Sends policy text + full HCPCS catalog to GPT-4o-mini with structured JSON output
2. Temperature=0 for near-deterministic results
3. Captures model version, prompt hash, and token usage in provenance artifacts

### `llm_mock` (no API key needed)

1. Regex extraction of code patterns from policy text
2. Cross-references against HCPCS catalog
3. Fixed confidence of 0.6
4. Useful for testing pipeline structure without API costs

## Handling Uncertainty

- **Confidence scores** are transparent: TF-IDF cosine similarity is directly used (0–1 range), with explicit boosts for literal matches
- **Thresholds** are configurable via `--threshold` to control the precision/recall tradeoff
- **Per-code provenance** lets consumers inspect *why* each code was inferred and at what confidence
- **Error isolation**: A single policy failure doesn't halt the pipeline — it produces an empty inference list and logs the error

## v2 Evolution Plan: Multi-Method Support

When Policybot adds RAG, ensemble, or other methods, the schema evolves additively without breaking v1 consumers.

### What Changes

```
PipelineRunMetadata gains (all optional):
  methods: list[InferenceMethodType]         # All methods used
  reconciliation_strategy: Optional[str]     # e.g., "weighted_average"

PolicyOutput gains (optional):
  method_results: list[MethodResult]         # Per-method raw outputs

New model:
  MethodResult:
    method: InferenceMethodType
    inferences: list[CodeInference]
    metadata: dict
```

### What Stays the Same

- `results[].inferences[]` remains the **final, reconciled** list of codes
- v1 consumers reading `inferences` continue to work unchanged
- `schema_version` bumps from `"1.0.0"` to `"2.0.0"` so consumers can detect the version

### v2 Output Example

```json
{
  "run_metadata": {
    "schema_version": "2.0.0",
    "methods": ["lexical_tfidf", "rag_embedding"],
    "reconciliation_strategy": "weighted_average"
  },
  "results": [{
    "policy_id": "...",
    "inferences": [{"code": "J3031", "confidence": 0.82, "...": "reconciled result"}],
    "method_results": [
      {"method": "lexical_tfidf", "inferences": [{"code": "J3031", "confidence": 0.55}]},
      {"method": "rag_embedding", "inferences": [{"code": "J3031", "confidence": 0.92}]}
    ]
  }]
}
```

### Adding a New Method

1. Create `src/policybot_hcpcs/inference/my_method.py`
2. Subclass `BaseInferenceMethod`
3. Decorate with `@register_method(InferenceMethodType.MY_METHOD)`
4. Add the enum value to `InferenceMethodType`
5. The registry, pipeline, and CLI pick it up automatically

## Design Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| TF-IDF as primary method | Low accuracy but deterministic, explainable, no API dependency |
| Pydantic v2 for schemas | Adds a dependency but provides validation, serialization, and self-documenting fields |
| Per-inference provenance | Higher storage cost per record, but enables full auditability |
| SHA-256 inputs_hash | Enables reproducibility checks without storing raw text in the audit trail |
| Strategy pattern + registry | Small upfront complexity, but makes adding methods trivial |
| Flat JSON output | Simple for consumers; v2 adds nested `method_results` only for multi-method runs |

## Evaluation

When `--labels` is provided, the pipeline computes:

- **Recall@k**: Fraction of true codes found in top-k predictions
- **Precision@k**: Fraction of top-k predictions that are correct
- **Coverage**: Fraction of policies with at least one correct code

Current v1 baseline (TF-IDF, top_k=10):
- recall@10: ~1.3%
- precision@10: ~2.8%
- coverage: ~7.5%

These numbers are expected for a simple TF-IDF baseline against implicit codes. The system is designed so that swapping in an LLM or RAG method improves accuracy without changing the schema or downstream consumers.
