"""Measure retrieval quality on a hand-written question set.

Usage (from the project root):
    python eval/run_eval.py --data data --questions eval/questions.json --k 4

Reports:
  hit@k          answerable questions where the right paper+page is in the top k
  refusal rate   unanswerable questions where nothing passes the similarity threshold
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from paperlens.embed import VectorIndex
from paperlens.ingest import pdf_to_chunks
from paperlens.retrieve import retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=0.25)
    args = ap.parse_args()

    index = VectorIndex()
    for pdf in sorted(pathlib.Path(args.data).glob("*.pdf")):
        index.add(pdf_to_chunks(pdf.read_bytes(), pdf.name))
    print(f"Indexed {len(index.chunks)} chunks")

    questions = json.loads(pathlib.Path(args.questions).read_text())
    hit = answerable = refused = unanswerable = 0
    for q in questions:
        if q["answerable"]:
            answerable += 1
            top = index.search(q["question"], args.k)
            ok = any(c.paper == q["paper"] and c.page == q["page"] for c, _ in top)
            hit += ok
            print(("HIT  " if ok else "MISS ") + q["question"])
        else:
            unanswerable += 1
            ok = not retrieve(index, q["question"], args.k, args.min_score)
            refused += ok
            print(("OK   " if ok else "WRONG") + " (unanswerable) " + q["question"])

    if answerable:
        print(f"\nhit@{args.k}: {hit}/{answerable} = {hit / answerable:.0%}")
    if unanswerable:
        print(f"refusal rate: {refused}/{unanswerable} = {refused / unanswerable:.0%}")


if __name__ == "__main__":
    main()
