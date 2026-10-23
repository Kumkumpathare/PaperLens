import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from paperlens.answer import answer
from paperlens.embed import VectorIndex
from paperlens.ingest import pdf_to_chunks
from paperlens.retrieve import retrieve

index = VectorIndex()
for pdf in sorted(pathlib.Path("data").glob("*.pdf")):
    index.add(pdf_to_chunks(pdf.read_bytes(), pdf.name))

questions = json.loads(pathlib.Path("eval/questions.json").read_text())
unanswerable = [q for q in questions if not q["answerable"]]

refused = 0
for q in unanswerable:
    hits = retrieve(index, q["question"], k=4, min_score=0.25)
    reply = answer(q["question"], hits)
    ok = "could not find" in reply.lower()
    refused += ok
    print(("OK   " if ok else "WRONG"), q["question"], "->", reply[:150].replace("\n", " "))

print(f"\nend-to-end refusal: {refused}/{len(unanswerable)}")