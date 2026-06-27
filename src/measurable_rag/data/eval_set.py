"""The abstention eval set: answerable (in-corpus) + unanswerable (out-of-corpus).

* Answerable questions are sampled from SciFact test claims that HAVE gold
  evidence in the corpus, so a correct system *can* answer them. (Reusing the
  labeled data keeps this slice honest — these are known-answerable.)
* Unanswerable questions are hand-curated to be plausibly biomedical yet not
  supported by the SciFact corpus. They are deliberately on-topic: an obviously
  off-topic question ("what is the capital of France?") is trivial to refuse and
  tests nothing. The hard case is a question that *sounds* answerable and pulls
  back related-but-insufficient chunks — tempting the model to confabulate.

Honest caveat: the unanswerable slice is hand-written, so an item or two could be
tangentially supported by some abstract. A more rigorous construction — holding
each answerable claim's gold documents out of the index and asking it anyway — is
noted as future work in the README.
"""
from __future__ import annotations

from .qrels import load_scifact_qrels, load_scifact_queries

# Plausibly biomedical, specific, and overstated — unlikely to be directly
# supported by SciFact abstracts, but close enough in topic to be hard to refuse.
UNANSWERABLE: list[str] = [
    "Does daily consumption of 80 grams of dark chocolate reverse coronary artery calcification?",
    "Can intermittent fasting for 16 hours a day cure rheumatoid arthritis?",
    "Does vitamin D3 at 10,000 IU daily eliminate all multiple sclerosis relapses?",
    "Is CRISPR editing of the PCSK9 gene an approved treatment for familial hypercholesterolemia in children?",
    "Does alkaline water at pH 9.5 prevent metastasis in stage IV breast cancer?",
    "Can a ketogenic diet fully replace insulin therapy in type 1 diabetes patients?",
    "Does transcranial magnetic stimulation restore vision in people with congenital blindness?",
    "Is psilocybin an FDA-approved first-line treatment for generalized anxiety disorder?",
    "Does daily turmeric supplementation reduce Alzheimer's plaque burden by 40 percent in humans?",
    "Can fecal microbiota transplantation cure type 2 diabetes within six months?",
    "Does sleeping exactly six hours a night maximize human lifespan compared to eight?",
    "Is there a vaccine that prevents every strain of the common cold?",
    "Does cold-water immersion at 10 degrees Celsius for 15 minutes daily reverse obesity?",
    "Can red-light therapy regrow lost permanent teeth in adult humans?",
    "Does a single monoclonal antibody dose permanently cure peanut allergy?",
]


def build_eval_set(n_answerable: int = 15) -> list[dict]:
    """Return a labeled list of {id, question, label} items.

    Answerable items are the first ``n_answerable`` SciFact test claims (sorted by
    id, so the set is deterministic) that have gold evidence; unanswerable items
    are the curated list above.
    """
    qrels = load_scifact_qrels(split="test")
    queries = load_scifact_queries()

    answerable_qids = [
        qid
        for qid in sorted(qrels, key=lambda q: int(q) if q.isdigit() else q)
        if qid in queries and any(score > 0 for score in qrels[qid].values())
    ]

    items: list[dict] = []
    for qid in answerable_qids[:n_answerable]:
        items.append({"id": f"scifact-{qid}", "question": queries[qid], "label": "answerable"})
    for i, question in enumerate(UNANSWERABLE):
        items.append({"id": f"unans-{i}", "question": question, "label": "unanswerable"})
    return items
