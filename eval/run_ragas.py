"""Ragas judged metrics (faithfulness, context precision, context recall) on a SUBSET of the eval results,
using the same OpenAI-compatible LLM as the app (e.g. Gemini) and the local embedding model.

Run from backend/ after a run of run_eval.py:
    pip install -r requirements-eval.txt
    python ../eval/run_ragas.py eval_results.jsonl --n 12

Notes
* Ragas makes several LLM calls per question per metric, so keep --n small on free tiers.
* Only answered questions with a reference answer are used. Contexts are the exact passages that
  were sent to the answering model (stored by run_eval.py), not truncated excerpts.
* Ragas' API changes between releases. If an import below fails, send the traceback.
* Small or "lite" judge models sometimes return malformed JSON; those cells come back as NaN.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from app.config import get_settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="eval_results.jsonl written by run_eval.py")
    ap.add_argument("--n", type=int, default=12, help="number of questions to judge")
    ap.add_argument("--out", default="ragas_scores.csv")
    args = ap.parse_args()

    s = get_settings()
    if not s.llm_base_url:
        sys.exit("Set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env (Ragas needs a judge LLM).")

    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    pool = [r for r in rows if r["status"] == "answered" and r["expected_answerable"]
            and r.get("reference") and r.get("contexts")]
    if not pool:
        sys.exit("No answered questions with references and contexts in that file.")
    if all(len(c) <= 400 for r in pool for c in r["contexts"]):
        print("WARNING: contexts look truncated (old results file). Re-run run_eval.py without --reuse first.\n")
    random.Random(0).shuffle(pool)
    sample = pool[: args.n]
    print(f"Judging {len(sample)} of {len(pool)} eligible questions with {s.llm_model}\n")

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=s.llm_model, base_url=s.llm_base_url, api_key=s.llm_api_key or "none",
        temperature=0, max_tokens=s.llm_max_tokens))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=s.embed_model))

    data = [{"user_input": r["question"], "retrieved_contexts": r["contexts"],
             "response": r["answer"], "reference": r["reference"]} for r in sample]
    result = evaluate(
        EvaluationDataset.from_list(data),
        metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm, embeddings=emb,
        run_config=RunConfig(max_workers=1, timeout=240),  # one at a time: free-tier rate limits
    )
    print("\n=== Ragas (mean over judged questions) ===")
    print(result)
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in sample])
    df.to_csv(args.out, index=False)
    print(f"\nPer-question scores saved to {args.out}")


if __name__ == "__main__":
    main()
