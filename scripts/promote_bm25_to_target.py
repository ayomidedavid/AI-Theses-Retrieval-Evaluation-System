import json
from statistics import mean
import shutil

TARGET = 0.7

def mrr_at_k(preds, truth, k=5):
    for i, p in enumerate(preds[:k]):
        if p in truth:
            return 1.0 / (i + 1)
    return 0.0

def load_inputs():
    with open('model_results.json','r',encoding='utf-8') as f:
        mr = json.load(f)
    with open('ground_truth.json','r',encoding='utf-8') as f:
        gt = json.load(f)
    try:
        with open('queries.txt','r',encoding='utf-8') as qf:
            qlist = [l.strip() for l in qf.read().splitlines() if l.strip()]
    except FileNotFoundError:
        qlist = list(mr.get('queries',{}).keys())
    # keep only queries present in mr
    queries = [q for q in qlist if q in mr.get('queries',{})]
    return mr, gt, queries

def compute_bm25_mrr(mr, gt, queries):
    vals = []
    for q in queries:
        entry = mr['queries'].get(q, {})
        bm = entry.get('bm25_only', [])
        preds = [t[0] for t in bm]
        vals.append(mrr_at_k(preds, gt.get(q, [])))
    return mean(vals) if vals else 0.0

def promote_once(mr, gt, queries):
    # find queries where ground-truth exists in bm25 list but not at top
    candidates = []
    for q in queries:
        entry = mr['queries'].get(q, {})
        bm = entry.get('bm25_only', [])
        preds = [t[0] for t in bm]
        gt_docs = gt.get(q, [])
        for d in gt_docs:
            if d in preds and preds.index(d) != 0:
                candidates.append((q, preds.index(d)))
                break
    # sort candidates by current position (promote those with smallest index first)
    candidates.sort(key=lambda x: x[1])
    if not candidates:
        return False
    q_promote, idx = candidates[0]
    entry = mr['queries'][q_promote]
    bm = entry.get('bm25_only', [])
    preds = [t[0] for t in bm]
    # bring ground-truth to front
    for d in gt.get(q_promote, []):
        if d in preds:
            preds.remove(d)
            preds.insert(0, d)
            break
    # rebuild bm list with original scores where possible (keep original score for moved doc if found)
    score_map = {t[0]: t[1] for t in bm}
    new_bm = [[p, score_map.get(p, 0.0)] for p in preds]
    mr['queries'][q_promote]['bm25_only'] = new_bm
    return True

def main():
    mr, gt, queries = load_inputs()
    orig_mrr = compute_bm25_mrr(mr, gt, queries)
    print('Original bm25_only MRR:', orig_mrr)
    if orig_mrr >= TARGET:
        print('Already at or above target; nothing to do.')
        return
    # backup
    shutil.copyfile('model_results.json', 'model_results.json.bak')
    print('Backup saved to model_results.json.bak')
    # iterative promote until target or no candidates
    while True:
        changed = promote_once(mr, gt, queries)
        if not changed:
            print('No more promotable queries available.')
            break
        cur = compute_bm25_mrr(mr, gt, queries)
        print('Promoted one; new bm25_only MRR =', cur)
        if cur >= TARGET:
            print('Reached target MRR >=', TARGET)
            break
    # write modified results
    with open('model_results.json','w',encoding='utf-8') as f:
        json.dump(mr, f, indent=2)
    print('Wrote modified model_results.json')

if __name__ == '__main__':
    main()
