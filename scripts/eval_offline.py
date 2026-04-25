"""
Offline evaluation script — runs the agent on each item in data/golden/golden_qa.jsonl
and computes aggregate metrics.

Metrics reported:
  Heuristic: citation_recall, keyword_hit_rate, groundedness_score,
             citation_coverage, tool_use_score
  LLM-as-judge (Part C): factuality, relevance, citation_quality, overall

Usage:
    python scripts/eval_offline.py
    python run.py --eval
"""
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when script is run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

GOLDEN_PATH = Path("data/golden/golden_qa.jsonl")
OUTPUTS_DIR = Path("outputs")


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    if not path.exists():
        console.print(f"[red]Golden dataset not found: {path}[/red]")
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def citation_recall(expected: list[str], actual: list[str]) -> float:
    """Fraction of expected citations that appear in actual citations."""
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if e in actual)
    return hits / len(expected)


def keyword_hit_rate(expected_keywords: list[str], answer: str) -> float:
    """Fraction of expected keywords (case-insensitive) present in the answer."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def main():
    from pathlib import Path as _Path
    from app.graph import compiled_graph
    from app.models import AgentOutput

    items = load_golden()
    if not items:
        return

    console.print(f"\n[bold]Running offline eval on {len(items)} golden items...[/bold]\n")

    results = []
    aggregate = {
        "citation_recall": [],
        "keyword_hit_rate": [],
        "groundedness_score": [],
        "citation_coverage": [],
        "tool_use_score": [],
        "llm_judge_factuality": [],
        "llm_judge_relevance": [],
        "llm_judge_citation": [],
        "llm_judge_overall": [],
    }

    for item in track(items, description="Evaluating..."):
        question = item["question"]
        expected_citations = item.get("expected_citations", [])
        expected_keywords = item.get("expected_keywords", [])

        start = time.time()
        state = compiled_graph.invoke({
            "question": question,
            "retrieved_docs": [],
            "context": "",
            "citations": [],
            "errors": [],
            "needs_clarification": False,
            "draft_answer": "",
            "final_answer": "",
            "start_time": start,
        })

        output_path = state.get("output_path", "")
        output_obj = None
        if output_path:
            try:
                data = json.loads(_Path(output_path).read_text(encoding="utf-8"))
                output_obj = AgentOutput(**data)
            except Exception:
                pass

        if not output_obj:
            results.append({"id": item.get("id"), "error": "no output"})
            continue

        answer = output_obj.final_answer
        actual_citations = output_obj.citations
        er = output_obj.eval_report

        cit_rec = citation_recall(expected_citations, actual_citations)
        kw_hit = keyword_hit_rate(expected_keywords, answer)

        result = {
            "id": item.get("id"),
            "question": question,
            "answer_snippet": answer[:120],
            "expected_citations": expected_citations,
            "actual_citations": actual_citations,
            "citation_recall": round(cit_rec, 3),
            "keyword_hit_rate": round(kw_hit, 3),
            "groundedness_score": er.groundedness_score,
            "citation_coverage": er.citation_coverage,
            "tool_use_score": er.tool_use_score,
            "llm_judge_factuality": er.llm_judge_factuality,
            "llm_judge_relevance": er.llm_judge_relevance,
            "llm_judge_citation": er.llm_judge_citation,
            "llm_judge_overall": er.llm_judge_overall,
            "llm_judge_reasoning": er.llm_judge_reasoning,
            "eval_notes": er.notes,
        }
        results.append(result)

        aggregate["citation_recall"].append(cit_rec)
        aggregate["keyword_hit_rate"].append(kw_hit)
        aggregate["groundedness_score"].append(er.groundedness_score)
        aggregate["citation_coverage"].append(er.citation_coverage)
        aggregate["tool_use_score"].append(er.tool_use_score)
        if er.llm_judge_overall > 0:
            aggregate["llm_judge_factuality"].append(er.llm_judge_factuality)
            aggregate["llm_judge_relevance"].append(er.llm_judge_relevance)
            aggregate["llm_judge_citation"].append(er.llm_judge_citation)
            aggregate["llm_judge_overall"].append(er.llm_judge_overall)

    # Compute means
    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {
        "n_items": len(items),
        "n_completed": len([r for r in results if "error" not in r]),
        "mean_citation_recall": mean(aggregate["citation_recall"]),
        "mean_keyword_hit_rate": mean(aggregate["keyword_hit_rate"]),
        "mean_groundedness_score": mean(aggregate["groundedness_score"]),
        "mean_citation_coverage": mean(aggregate["citation_coverage"]),
        "mean_tool_use_score": mean(aggregate["tool_use_score"]),
        "mean_llm_judge_factuality": mean(aggregate["llm_judge_factuality"]),
        "mean_llm_judge_relevance": mean(aggregate["llm_judge_relevance"]),
        "mean_llm_judge_citation": mean(aggregate["llm_judge_citation"]),
        "mean_llm_judge_overall": mean(aggregate["llm_judge_overall"]),
    }

    # Print summary table
    heuristic_keys = {"citation_recall", "keyword_hit_rate", "groundedness_score",
                      "citation_coverage", "tool_use_score"}
    judge_keys = {"llm_judge_factuality", "llm_judge_relevance",
                  "llm_judge_citation", "llm_judge_overall"}

    table = Table(title="Offline Eval Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean Score", justify="right")
    for k, v in summary.items():
        if k.startswith("mean_") and k[5:] in heuristic_keys:
            table.add_row(k[5:], f"{v:.3f}")
    table.add_section()
    for k, v in summary.items():
        if k.startswith("mean_") and k[5:] in judge_keys:
            label = k[5:].replace("llm_judge_", "judge_")
            table.add_row(label, f"{v:.3f}")
    console.print(table)
    console.print(f"\n[dim]Items: {summary['n_items']} | Completed: {summary['n_completed']}[/dim]")

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"eval_offline_{ts}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"[green]Results saved to: {out_path}[/green]")


if __name__ == "__main__":
    main()
