"""
evaluate_retrieval.py — Evaluator for Hybrid Retriever
=======================================================

Metrics: Precision@K, Recall@K, MRR, NDCG@K

Usage:
    python evaluate_retrieval.py --domain law --k 1 3 5
    python evaluate_retrieval.py --domain culture --k 1 3 5
    python evaluate_retrieval.py --domain law --dataset evaluator.json
    python evaluate_retrieval.py --domain law --stage hybrid   # skip reranker, test hybrid scoring alone
    python evaluate_retrieval.py --domain law --stage full     # hybrid + reranker (default)
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Set

from vector import (
    retriever,
    get_canonical_id,
    retrieve_candidates,
)


# ─────────────────────────────
# DATASET LOADER
# ─────────────────────────────
EVAL_DATASET_FALLBACK = {
    "law": {
        "queries": [
            "What is the maximum imprisonment for a person who commits murder?",
            "Who has the right to file a complaint if the victim of a crime is under 16 years old?",
            "What are the types of principal sentences according to Article 65?",
            "How many years of probation shall judges impose when applying capital punishment?",
            "What is the purpose of sentencing according to Article 51?",
            "Which entities are considered subjects of crimes under Article 45?",
            "Until when can a complaint for a crime be withdrawn?",
            "How is the criminal sentence affected if murder is committed against the offender's parents or spouse?",
            "How is the term 'Person' defined in Article 145?",
            "What types of goods may be confiscated as an additional sentence under Article 91?",
            "What factors must be considered when sentencing a Corporation?",
            "What actions constitute a preparation to commit a crime according to Article 15?",
            "What is the penalty for negligence that results in an electrical grid being damaged?",
            "What is the penalty for inviting Indonesian citizens to join a foreign army without a permit?",
            "What is the maximum imprisonment for habitual fencing under Article 592?",
        ],
        "relevance": {
            "What is the maximum imprisonment for a person who commits murder?": ["penal_Article_458"],
            "Who has the right to file a complaint if the victim of a crime is under 16 years old?": ["penal_Article_25"],
            "What are the types of principal sentences according to Article 65?": ["penal_Article_65"],
            "How many years of probation shall judges impose when applying capital punishment?": ["penal_Article_100"],
            "What is the purpose of sentencing according to Article 51?": ["penal_Article_51"],
            "Which entities are considered subjects of crimes under Article 45?": ["penal_Article_45"],
            "Until when can a complaint for a crime be withdrawn?": ["penal_Article_25"],
            "How is the criminal sentence affected if murder is committed against the offender's parents or spouse?": ["penal_Article_458"],
            "How is the term 'Person' defined in Article 145?": ["penal_Article_145"],
            "What types of goods may be confiscated as an additional sentence under Article 91?": ["penal_Article_91"],
            "What factors must be considered when sentencing a Corporation?": ["penal_Article_56"],
            "What actions constitute a preparation to commit a crime according to Article 15?": ["penal_Article_15"],
            "What is the penalty for negligence that results in an electrical grid being damaged?": ["penal_Article_320"],
            "What is the penalty for inviting Indonesian citizens to join a foreign army without a permit?": ["penal_Article_201"],
            "What is the maximum imprisonment for habitual fencing under Article 592?": ["penal_Article_592"],
        }
    },
    "culture": {
        "queries": [
            "What is the main purpose of the Nyepi day in Bali?",
            "When does the Galungan ceremony take place?",
            "Why are tourists not allowed to step on Canang Sari offerings?",
            "How do Balinese Hindus purify sacred objects before Nyepi?",
            "What happens 10 days after Galungan?",
            "Who is Dewi Saraswati?",
            "What is the meaning behind the Pagerwesi ceremony?",
            "Why must tourists wear a sarong and sash in temples?",
            "What is the Ngaben ceremony?",
            "When is the Ngarorasin ceremony performed?",
            "What does the Metatah ceremony involve?",
            "Why is it forbidden to point your feet at a priest in Bali?",
            "What is the consequence of flying a drone over an active temple ceremony?",
            "How often does the Otonan ceremony occur?",
            "What is the purpose of the Tumpek Kandang ceremony?",
            "Why should tourists not haggle aggressively with elderly street vendors?",
            "What is the Mediksha ceremony?",
            "When does the Tawur Kesanga ceremony take place?",
            "What should you do if you accidentally step on a Canang Sari?",
            "Why are menstruating women not allowed to enter temples?",
            "What is the Pawiwohan ceremony?",
            "How does the Nyekah ceremony help the ancestor's soul?",
            "What is the penalty for touching a Balinese person's head?",
            "Why is it taboo to swim in areas marked with red flags?",
            "What is the Tumpek Bubuh ceremony about?",
            "What happens if a tourist disrespects the Pecalang?",
            "What is the Magedong-Gedongan ceremony?",
            "Why must tourists use their right hand for transactions?",
            "What is the Dapetan ceremony?",
            "Why is it forbidden to dump plastic into rivers in Bali?",
        ],
        "relevance": {
            "What is the main purpose of the Nyepi day in Bali?": ["ceremony_3"],
            "When does the Galungan ceremony take place?": ["ceremony_1"],
            "Why are tourists not allowed to step on Canang Sari offerings?": ["taboo_11"],
            "How do Balinese Hindus purify sacred objects before Nyepi?": ["ceremony_4"],
            "What happens 10 days after Galungan?": ["ceremony_2"],
            "Who is Dewi Saraswati?": ["ceremony_5"],
            "What is the meaning behind the Pagerwesi ceremony?": ["ceremony_6"],
            "Why must tourists wear a sarong and sash in temples?": ["taboo_2"],
            "What is the Ngaben ceremony?": ["ceremony_8"],
            "When is the Ngarorasin ceremony performed?": ["ceremony_9"],
            "What does the Metatah ceremony involve?": ["ceremony_16"],
            "Why is it forbidden to point your feet at a priest in Bali?": ["taboo_22"],
            "What is the consequence of flying a drone over an active temple ceremony?": ["taboo_32"],
            "How often does the Otonan ceremony occur?": ["ceremony_15"],
            "What is the purpose of the Tumpek Kandang ceremony?": ["ceremony_20"],
            "Why should tourists not haggle aggressively with elderly street vendors?": ["taboo_49"],
            "What is the Mediksha ceremony?": ["ceremony_22"],
            "When does the Tawur Kesanga ceremony take place?": ["ceremony_19"],
            "What should you do if you accidentally step on a Canang Sari?": ["taboo_11"],
            "Why are menstruating women not allowed to enter temples?": ["taboo_1"],
            "What is the Pawiwohan ceremony?": ["ceremony_17"],
            "How does the Nyekah ceremony help the ancestor's soul?": ["ceremony_10"],
            "What is the penalty for touching a Balinese person's head?": ["taboo_21"],
            "Why is it taboo to swim in areas marked with red flags?": ["taboo_50"],
            "What is the Tumpek Bubuh ceremony about?": ["ceremony_21"],
            "What happens if a tourist disrespects the Pecalang?": ["taboo_33"],
            "What is the Magedong-Gedongan ceremony?": ["ceremony_12"],
            "Why must tourists use their right hand for transactions?": ["taboo_23"],
            "What is the Dapetan ceremony?": ["ceremony_13"],
            "Why is it forbidden to dump plastic into rivers in Bali?": ["taboo_45"],
        }
    }
}


def load_dataset(path: str, domain: str) -> dict:
    """Load dataset from JSON or use fallback."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle nested structure from evaluator.json
            if "datasets" in data and domain in data["datasets"]:
                return data["datasets"][domain]
            return data

    print(f"⚠️  '{path}' not found, using built-in fallback.")
    return EVAL_DATASET_FALLBACK.get(domain, {"queries": [], "relevance": {}})


# ─────────────────────────────
# RETRIEVAL WRAPPER
# ─────────────────────────────
def get_ranked_docs(query: str, domain: str, max_k: int, stage: str = "full"):
    """
    Run retrieval pipeline and return ranked Document objects.

    stage="hybrid" -> stop right after hybrid BM25+vector fusion (no reranker).
    stage="full"    -> hybrid fusion -> cross-encoder reranker (default,
                       matches the real production pipeline).
    """
    vectorstore = retriever.vectorstores[domain]
    bm25 = retriever.bm25[domain]

    # retrieve_candidates() returns a Dict[canonical_id, Document]
    hybrid_dict = retrieve_candidates(vectorstore, bm25, query, domain)
    candidates = list(hybrid_dict.values())

    if stage == "hybrid":
        # Simple fallback sort if testing hybrid without reranker
        candidates.sort(
            key=lambda d: d.metadata.get("bm25_score", 0), 
            reverse=True
        )
        return candidates[:max_k]

    # full pipeline: feed the candidates into the reranker
    reranked = retriever.reranker.rerank(query, candidates, top_k=max_k)

    return reranked


# ─────────────────────────────
# METRICS
# ─────────────────────────────
def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for rid in ranked[:k] if rid in relevant) / k


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for rid in ranked[:k] if rid in relevant) / len(relevant)


def reciprocal_rank(ranked: List[str], relevant: Set[str]) -> float:
    for idx, rid in enumerate(ranked, start=1):
        if rid in relevant:
            return 1.0 / idx
    return 0.0


def dcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(ranked[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def idcg_at_k(relevant: Set[str], k: int) -> float:
    n = min(len(relevant), k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, n + 1))


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    idcg = idcg_at_k(relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


# ─────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────
def evaluate(dataset: dict, domain: str, k_values: List[int], max_retrieve: int, stage: str):
    queries = dataset["queries"]
    relevance_map = dataset["relevance"]

    per_query_rows = []
    agg: Dict[str, List[float]] = {f"precision@{k}": [] for k in k_values}
    agg.update({f"recall@{k}": [] for k in k_values})
    agg.update({f"ndcg@{k}": [] for k in k_values})
    agg["mrr"] = []

    max_k = max(k_values + [max_retrieve])

    for q in queries:
        relevant = set(relevance_map.get(q, []))
        ranked_docs = get_ranked_docs(q, domain, max_k, stage=stage)
        ranked_ids = [get_canonical_id(d) for d in ranked_docs]
        
        # Ambil teks hasil (cross encoder / final rank)
        ranked_texts = "\n\n---\n\n".join([d.page_content.strip() for d in ranked_docs])

        row = {
            "query": q,
            "relevant_ids": ";".join(sorted(relevant)),
            "retrieved_ids": ";".join(ranked_ids),
            "retrieved_texts": ranked_texts,
        }

        rr = reciprocal_rank(ranked_ids, relevant)
        row["mrr"] = round(rr, 4)
        agg["mrr"].append(rr)

        # ====== PRINT KE TERMINAL ======
        print(f"\n🔹 QUERY   : {q}")
        print(f"🔸 EXPECTED: {', '.join(sorted(relevant))}")
        
        if not ranked_docs:
            print("   (Tidak ada dokumen ditemukan)")
            
        for i, doc in enumerate(ranked_docs):
            cid = get_canonical_id(doc)
            preview = doc.page_content.replace('\n', ' ').strip()[:100]
            marker = "✅" if cid in relevant else "❌"
            print(f"   {i+1}. {marker} [{cid}] {preview}...")
        # ===============================

        for k in k_values:
            p = precision_at_k(ranked_ids, relevant, k)
            r = recall_at_k(ranked_ids, relevant, k)
            n = ndcg_at_k(ranked_ids, relevant, k)

            row[f"precision@{k}"] = round(p, 4)
            row[f"recall@{k}"] = round(r, 4)
            row[f"ndcg@{k}"] = round(n, 4)

            agg[f"precision@{k}"].append(p)
            agg[f"recall@{k}"].append(r)
            agg[f"ndcg@{k}"].append(n)

        per_query_rows.append(row)

    summary = {
        metric: round(sum(values) / len(values), 4) if values else 0.0
        for metric, values in agg.items()
    }

    return per_query_rows, summary


def save_csv(rows: List[dict], path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluator Hybrid Retriever (Precision@K, Recall@K, MRR, NDCG@K)"
    )
    parser.add_argument("--domain", default="law", choices=["law", "culture"],
                        help="Domain to evaluate (default: law)")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5],
                        help="K values for metrics (default: 1 3 5)")
    parser.add_argument("--max-retrieve", type=int, default=5,
                        help="Max docs to retrieve per query (default: 5)")
    parser.add_argument("--dataset", type=str, default="evaluator.json",
                        help="Path to evaluation dataset JSON (default: evaluator.json)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Output folder for CSVs (default: current dir)")
    parser.add_argument("--stage", type=str, default="full", choices=["hybrid", "full"],
                        help="'hybrid' = BM25+vector fusion only (no reranker). "
                             "'full' = hybrid + cross-encoder reranker (default).")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.domain)

    print(f"\n🔎 Evaluating domain='{args.domain}' | stage='{args.stage}' | "
          f"K={args.k} | queries={len(dataset['queries'])}\n")

    per_query_rows, summary = evaluate(dataset, args.domain, args.k, args.max_retrieve, args.stage)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.stage}"
    
    # Ekstrak nama dataset untuk penamaan file CSV yang unik
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    if dataset_name == "evaluator": # Fallback jika pakai file default lama
        dataset_name = ""
    else:
        dataset_name = f"_{dataset_name}"
        
    per_query_path = os.path.join(args.out_dir, f"per_query_{args.domain}{dataset_name}{suffix}.csv")
    summary_path = os.path.join(args.out_dir, f"summary_{args.domain}{dataset_name}{suffix}.csv")

    save_csv(per_query_rows, per_query_path)
    save_csv([summary], summary_path)

    print(f"📄 Per-query metrics  → {per_query_path}")
    print(f"📄 Summary metrics    → {summary_path}")
    print("\n" + "=" * 50)
    print(f"SUMMARY — {args.domain.upper()} DOMAIN ({args.stage})")
    print("=" * 50)
    for metric, value in summary.items():
        print(f"{metric:18s}: {value}")


if __name__ == "__main__":
    main()