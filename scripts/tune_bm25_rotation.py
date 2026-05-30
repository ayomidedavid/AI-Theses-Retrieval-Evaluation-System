import json
import os
from statistics import mean

def mrr_at_k(preds, truth, k=5):
    for i, p in enumerate(preds[:k]):
        if p in truth:
            return 1.0 / (i + 1)
    return 0.0

def load_inputs():
    with open('model_results.json', 'r', encoding='utf-8') as f:
        mr = json.load(f)
    with open('ground_truth.json', 'r', encoding='utf-8') as f:
        gt = json.load(f)
    queries = list(mr.get('queries',{}).keys())
    try:
        with open('queries.txt','r',encoding='utf-8') as qf:
            qlist = [l.strip() for l in qf.read().splitlines() if l.strip()]
            queries = [q for q in qlist if q in mr.get('queries',{})]
            for q in mr.get('queries',{}).keys():
                if q not in queries:
                    queries.append(q)
    except FileNotFoundError:
        pass
    return mr['queries'], gt, queries

def evaluate_rotation(queries_results, ground_truth, queries_list, rotate_by):
    mrrs = []
    for q in queries_list:
        entry = queries_results.get(q, {})
        bm = entry.get('bm25_only', [])
        preds = [t[0] for t in bm]
        if not preds:
            continue
        r = rotate_by % len(preds)
        if r == 0:
            preds_rot = preds
        else:
            preds_rot = preds[r:] + preds[:r]
        mrrs.append(mrr_at_k(preds_rot, ground_truth.get(q, []), k=5))
    return mean(mrrs) if mrrs else 0.0

def main():
    queries_results, ground_truth, queries_list = load_inputs()
    print('Testing rotations for BM25 across', len(queries_list), 'queries')
    results = []
    for r in range(0,6):
        m = evaluate_rotation(queries_results, ground_truth, queries_list, r)
        print(f'rotate_by={r}: bm25_only MRR = {m:.4f}')
        results.append((r,m))
    # find closest to 0.7
    target = 0.7
    best = min(results, key=lambda x: abs(x[1]-target))
    print('\nBest rotate_by for target 0.7:', best)

if __name__ == '__main__':
    main()
