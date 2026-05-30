"""Simple learning-to-rank script (pointwise logistic regression).

Usage: python learn_to_rank.py

Reads `model_results.json` and `ground_truth.json` from the repo root,
builds per-query candidate feature vectors from TF-IDF and BM25 scores,
performs leave-one-query-out training of a logistic regression ranker,
and writes `learn_to_rank_results.json` with per-query predictions and
overall mean MRR / P@5.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression


def mrr_at_k(preds, truth, k=5):
    for i, p in enumerate(preds[:k]):
        if p in truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(preds, truth, k=5):
    return sum(1 for p in preds[:k] if p in truth) / float(k)


def load_data():
    with open('model_results.json', 'r', encoding='utf-8') as f:
        mr = json.load(f)
    with open('ground_truth.json', 'r', encoding='utf-8') as f:
        gt = json.load(f)
    return mr['queries'], gt


def build_candidate_features(entry):
    # entry contains lists: 'tfidf_only', 'bm25_only', and hybrid dicts
    cand = set()
    tf = entry.get('tfidf_only', [])
    bm = entry.get('bm25_only', [])
    hy_no = entry.get('hybrid_no_optimizer', {}).get('top5', [])
    hy_opt = entry.get('hybrid_with_optimizer', {}).get('top5', [])
    for t in tf: cand.add(t[0])
    for b in bm: cand.add(b[0])
    for h in hy_no: cand.add(h[0])
    for h in hy_opt: cand.add(h[0])

    tf_dict = {t[0]: t[1] for t in tf}
    bm_dict = {b[0]: b[1] for b in bm}
    hy_no_dict = {h[0]: h[1] for h in hy_no}
    hy_opt_dict = {h[0]: h[1] for h in hy_opt}

    rows = []
    for d in cand:
        rows.append((d,
                     float(tf_dict.get(d, 0.0)),
                     float(bm_dict.get(d, 0.0)),
                     float(hy_no_dict.get(d, 0.0)),
                     float(hy_opt_dict.get(d, 0.0))))
    return rows


def normalize_features(rows):
    # rows: list of (doc, tf, bm, hy_no, hy_opt)
    if len(rows) == 0:
        return rows, []
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
    # per-column max normalization
    maxs = arr.max(axis=0)
    maxs[maxs == 0] = 1.0
    arr_norm = arr / maxs
    norm_rows = []
    for i, r in enumerate(rows):
        norm_rows.append((r[0],) + tuple(arr_norm[i].tolist()))
    return norm_rows, maxs.tolist()


def main():
    queries_data, gt = load_data()

    # prepare valid queries (have ground truth)
    valid_queries = [q for q in queries_data.keys() if q in gt and len(gt[q])>0]
    if len(valid_queries) == 0:
        print('No valid queries with ground-truth found.')
        return

    per_query_candidates = {}
    for q in queries_data:
        rows = build_candidate_features(queries_data[q])
        norm_rows, _ = normalize_features(rows)
        per_query_candidates[q] = norm_rows

    results = {}
    mrrs = []
    p5s = []

    for test_q in valid_queries:
        # build train set from all other queries
        X_train = []
        y_train = []
        for q in valid_queries:
            if q == test_q: continue
            for doc, a, b, c, d in per_query_candidates.get(q, []):
                X_train.append([a,b,c,d])
                y_train.append(1 if doc in gt.get(q, []) else 0)

        # If training set empty, skip
        if len(X_train) == 0:
            continue

        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=int)

        # train logistic regression
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)

        # predict on test candidates
        test_rows = per_query_candidates.get(test_q, [])
        docs = [r[0] for r in test_rows]
        X_test = np.array([[r[1],r[2],r[3],r[4]] for r in test_rows], dtype=float)
        if X_test.shape[0] == 0:
            continue
        probs = clf.predict_proba(X_test)[:,1]
        idx = np.argsort(probs)[::-1]
        ranked = [docs[i] for i in idx]

        mrr = mrr_at_k(ranked, gt[test_q], k=5)
        p5 = precision_at_k(ranked, gt[test_q], k=5)
        mrrs.append(mrr)
        p5s.append(p5)

        results[test_q] = {
            'predicted_top5': [(ranked[i], float(probs[idx[i]])) for i in range(min(5,len(ranked)))],
            'MRR': mrr,
            'P5': p5
        }

    mean_mrr = float(np.mean(mrrs)) if len(mrrs)>0 else 0.0
    mean_p5 = float(np.mean(p5s)) if len(p5s)>0 else 0.0

    # baselines
    def baseline_score(model_key):
        vals = []
        for q in valid_queries:
            entry = queries_data[q]
            lst = []
            if model_key == 'tfidf': lst = [t[0] for t in entry.get('tfidf_only', [])]
            if model_key == 'bm25': lst = [t[0] for t in entry.get('bm25_only', [])]
            if len(lst)==0:
                continue
            vals.append(mrr_at_k(lst, gt[q], k=5))
        return float(np.mean(vals)) if len(vals)>0 else 0.0

    baseline_tfidf = baseline_score('tfidf')
    baseline_bm25 = baseline_score('bm25')

    out = {
        'mean_MRR_learned': mean_mrr,
        'mean_P5_learned': mean_p5,
        'baseline_tfidf_MRR': baseline_tfidf,
        'baseline_bm25_MRR': baseline_bm25,
        'per_query': results
    }

    with open('learn_to_rank_results.json','w',encoding='utf-8') as f:
        json.dump(out,f,indent=2)

    print('Wrote learn_to_rank_results.json')
    print('Mean MRR (learned):', mean_mrr)
    print('Mean P@5 (learned):', mean_p5)
    print('Baseline TF-IDF MRR:', baseline_tfidf)
    print('Baseline BM25 MRR:', baseline_bm25)


if __name__ == '__main__':
    main()
