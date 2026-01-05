# claim-conditioned-cad

## Model Overview (What Runs by Default)

- Retriever embedder: `intfloat/e5-large-v2` (configurable via `embed_model`).
- Generator (open-weight): `meta-llama/Llama-2-13b-chat-hf` (configurable via `model_name` or `--model`).
- Entailment verifier: `roberta-large-mnli` (configurable via `nli_model`, can be `mock`).
- API mode: uses a stub client unless you implement `APIClient` in `cc_cad/decoding/cad.py`.

## Required Commands (End-to-End)

### 1) Create environment + install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers sentence-transformers faiss-cpu datasets pyyaml pandas matplotlib pytest huggingface_hub
```

### 2) Download Wikipedia paragraphs (parquet-based)

```bash
python scripts/download_kilt_wikipedia.py --output data/wiki_paragraphs.jsonl --language 20231101.en --max_docs 20000 --hard_exit
```

### 3) Build FAISS index

```bash
python -c "from cc_cad.retrieval.faiss_index import FaissIndex; idx=FaissIndex(model_name='intfloat/e5-large-v2'); idx.build_from_jsonl('data/wiki_paragraphs.jsonl'); idx.save('results/wiki_index')"
```

### 4) Run tests

```bash
pytest cc_cad/tests
```

### 5) Run the full sweep

```bash
python scripts/sweep_and_report.py
```

## GPU note

For the full sweep with large open-weight models, run on a machine with an appropriate GPU and use `--device cuda`.

## Open-weight mode (single run)

```bash
python -m cc_cad.runners.run_experiment \
  --dataset nq \
  --mode open \
  --method cc_cad \
  --config configs/nq.yaml \
  --max_examples 50 \
  --output_dir results/quickstart
```

## API mode

Implement `APIClient` in `cc_cad/decoding/cad.py` with:
- `generate_with_context(prompt, evidence)`
- `generate_without_context(prompt)`

Then run:

```bash
python -m cc_cad.runners.run_experiment \
  --dataset hotpotqa \
  --mode api \
  --method cc_cad \
  --config configs/hotpotqa.yaml \
  --output_dir results/hotpotqa_api
```

## Outputs

Each run writes:
- `predictions.jsonl`: answers, claims, citations
- `results.jsonl`: per-example metrics
- `metrics.json`: aggregate metrics
- `trace.jsonl`: per-claim entailment and citations

## Tables and figures

```bash
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/summarize_human_eval.py
```

Tables are written to `results/tables.md` and the human eval template to `results/human_eval/items.csv`.

## Configs

Edit `configs/*.yaml` to set:
- `wiki_path`: path to the wikipedia jsonl
- `index_dir`: FAISS index directory
- `embed_model`: sentence-transformers model
- `nli_model`: NLI model name (use `mock` for heuristic testing)
- `model_name`: open-weight LLM
