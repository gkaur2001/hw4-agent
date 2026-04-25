"""
CLI entrypoint for the HW4 Starter Agent.

Usage:
    python run.py --question "How do late submissions work?"
    python run.py --eval          # run offline evaluation
    python run.py --question "..." --quiet   # suppress rich formatting
"""
import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_question(question: str, quiet: bool = False) -> dict:
    """Run the agent on a single question and return the output dict."""
    from app.graph import compiled_graph
    from app.models import AgentOutput

    start = time.time()
    initial_state = {
        "question": question,
        "retrieved_docs": [],
        "context": "",
        "citations": [],
        "errors": [],
        "needs_clarification": False,
        "draft_answer": "",
        "final_answer": "",
        "start_time": start,
    }

    final_state = compiled_graph.invoke(initial_state)

    output_path = final_state.get("output_path", "")
    errors = final_state.get("errors", [])

    # Load the validated output from the JSON file written by finalize_node
    output_data: dict = {}
    output_obj = None
    if output_path:
        try:
            output_data = json.loads(Path(output_path).read_text(encoding="utf-8"))
            output_obj = AgentOutput(**output_data)
        except Exception as exc:
            console.print(f"[red]Could not load output file: {exc}[/red]")

    if not quiet and output_obj:
        _print_summary(output_obj, output_path, errors)
    elif not quiet:
        console.print(f"[yellow]No output produced. Errors: {errors}[/yellow]")

    return output_data


def _print_summary(output, output_path: str, errors: list):
    console.print()
    console.print(Panel(f"[bold]{output.question}[/bold]", title="Question", border_style="blue"))
    console.print()
    console.print(Panel(output.final_answer, title="Answer", border_style="green"))

    if output.citations:
        console.print(f"\n[bold]Citations:[/bold] {', '.join(output.citations)}")

    # Reasoning trace
    rt = output.reasoning_trace
    if rt.plan:
        rtable = Table(title="Reasoning Trace", show_header=True, header_style="bold cyan")
        rtable.add_column("Step", style="dim", width=4)
        rtable.add_column("Plan")
        for i, step in enumerate(rt.plan, 1):
            rtable.add_row(str(i), step)
        if rt.decision:
            rtable.add_section()
            rtable.add_row("→", f"[italic]{rt.decision}[/italic]")
        console.print(rtable)

    # Eval report table
    er = output.eval_report
    table = Table(title="Eval Report", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_row("Groundedness", f"{er.groundedness_score:.2f}")
    table.add_row("Citation coverage", f"{er.citation_coverage:.2f}")
    table.add_row("Tool-use", f"{er.tool_use_score:.2f}")
    if er.llm_judge_overall > 0:
        table.add_section()
        table.add_row("Judge — factuality", f"{er.llm_judge_factuality:.2f}")
        table.add_row("Judge — relevance", f"{er.llm_judge_relevance:.2f}")
        table.add_row("Judge — citation", f"{er.llm_judge_citation:.2f}")
        table.add_row("[bold]Judge — overall[/bold]", f"[bold]{er.llm_judge_overall:.2f}[/bold]")
    console.print(table)
    if er.llm_judge_reasoning and er.llm_judge_reasoning not in ("skipped", ""):
        console.print(f"[dim]Judge reasoning:[/dim] {er.llm_judge_reasoning}")
    if er.notes and er.notes != "ok":
        console.print(f"[yellow]Notes:[/yellow] {er.notes}")

    console.print(f"\n[dim]Latency: {output.run_meta.latency_ms} ms | Model: {output.run_meta.model}[/dim]")
    console.print(f"[dim]Output saved to: {output_path}[/dim]")

    if errors:
        console.print(f"\n[red]Errors:[/red] {errors}")


def main():
    parser = argparse.ArgumentParser(description="HW4 Starter Agent CLI")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question")
    parser.add_argument("--eval", action="store_true", help="Run offline evaluation")
    parser.add_argument("--quiet", action="store_true", help="Suppress rich output")
    args = parser.parse_args()

    if args.eval:
        import scripts.eval_offline as eval_script
        eval_script.main()
        return

    if args.question:
        run_question(args.question, quiet=args.quiet)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
