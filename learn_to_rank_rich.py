"""Learn-to-rank with richer features derived from filenames and available scores.

Features (constructed from available workspace data):
- title tokens overlap (query vs filename)
- normalized TF-IDF score (if available)
- normalized BM25 score (if available)
- BM25 - TFIDF difference
- title length (token count)
- year found in filename (binary)
- semantic similarity between query and title (SBERT if installed, else TF-IDF)

Note: This script cannot extract title/abstract/full-text features unless PDFs
are present in `backend/downloaded_pdfs/theses/`. It uses filenames from
`model_results.json` as proxies.
"""
import json
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def title_from_filename(fname):
    return re.sub(r"\.pdf$", "", fname, flags=re.IGNORECASE)


def year_in_text(text):
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return int(m.group(0)) if m else None


def build_candidates(entry):
    cand = set()
    for k in ['tfidf_only', 'bm25_only']:
        for t in entry.get(k, []):
            cand.add(t[0])
    # include hybrid lists if present
    for hkey in ['hybrid_no_optimizer', 'hybrid_with_optimizer']:
        v = entry.get(hkey)
        if isinstance(v, dict):
            for t in v.get('top5', []):
                cand.add(t[0])
    return list(cand)


def semantic_sim_query_title(query, titles, use_sbert=False):
    # titles: list of title strings
    if use_sbert:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            qv = model.encode([query])[0]
            tv = model.encode(titles)
            sims = cosine_similarity([qv], tv)[0].tolist()
            return sims
        except Exception:
            use_sbert = False
    # fallback: TF-IDF on titles+query
    vec = TfidfVectorizer().fit([query] + titles)
    mat = vec.transform([query] + titles)
    qvec = mat[0]
    tmat = mat[1:]
    sims = cosine_similarity(qvec, tmat)[0].tolist()
    return sims


def main():
    with open('model_results.json','r',encoding='utf-8') as f:
        mr = json.load(f)
    with open('ground_truth.json','r',encoding='utf-8') as f:
        gt = json.load(f)

    queries = list(mr['queries'].keys())
    valid_queries = [q for q in queries if q in gt and len(gt[q])>0]
    if not valid_queries:
        print('No valid queries with ground truth.')
        return

    # Build per-query candidate features
    per_query_feats = {}
    for q in mr['queries']:
        entry = mr['queries'][q]
        cand = build_candidates(entry)
        titles = [title_from_filename(c) for c in cand]
        # collect raw scores
        tf_map = {t[0]:t[1] for t in entry.get('tfidf_only', [])}
        bm_map = {b[0]:b[1] for b in entry.get('bm25_only', [])}

        # semantic sims (try SBERT)
        sims = semantic_sim_query_title(q, titles, use_sbert=True)

        # assemble rows
        rows = []
        for i, doc in enumerate(cand):
            title = titles[i]
            t_tokens = set(tokenize(title))
            q_tokens = set(tokenize(q))
            overlap = len(t_tokens.intersection(q_tokens)) / float(max(1, len(t_tokens.union(q_tokens))))
            tf_raw = float(tf_map.get(doc, 0.0))
            bm_raw = float(bm_map.get(doc, 0.0))
            title_len = len(t_tokens)
            y = year_in_text(title)
            year_flag = 1 if y is not None else 0
            sem_sim = float(sims[i]) if i < len(sims) else 0.0
            rows.append({
                'doc': doc,
                'title': title,
                'overlap': overlap,
                'tf_raw': tf_raw,
                'bm_raw': bm_raw,
                'title_len': title_len,
                'year_flag': year_flag,
                'sem_sim': sem_sim
            })
        # normalize tf and bm by max in this candidate set
        if rows:
            max_tf = max(r['tf_raw'] for r in rows) or 1.0
            max_bm = max(r['bm_raw'] for r in rows) or 1.0
            for r in rows:
                r['tf_norm'] = r['tf_raw'] / max_tf
                r['bm_norm'] = r['bm_raw'] / max_bm
                r['bm_minus_tf'] = r['bm_norm'] - r['tf_norm']

        per_query_feats[q] = rows

    # Train leave-one-query-out logistic regressor on these features
    results = {}
    mrrs = []
    p5s = []

    for test_q in valid_queries:
        # prepare training data
        X_train = []
        y_train = []
        for q in valid_queries:
            if q == test_q: continue
            for r in per_query_feats.get(q, []):
                X_train.append([r['tf_norm'], r['bm_norm'], r['bm_minus_tf'], r['overlap'], r['title_len'], r['year_flag'], r['sem_sim']])
                y_train.append(1 if r['doc'] in gt.get(q, []) else 0)
        if not X_train:
            continue
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=int)

        clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)

        # test
        test_rows = per_query_feats.get(test_q, [])
        docs = [r['doc'] for r in test_rows]
        X_test = np.array([[r['tf_norm'], r['bm_norm'], r['bm_minus_tf'], r['overlap'], r['title_len'], r['year_flag'], r['sem_sim']] for r in test_rows], dtype=float)
        if X_test.shape[0] == 0:
            continue
        probs = clf.predict_proba(X_test)[:,1]
        idx = np.argsort(probs)[::-1]
        ranked = [docs[i] for i in idx]

        # metrics
        def mrr_at_5(preds, truth):
            for i,p in enumerate(preds[:5]):
                if p in truth:
                    return 1.0/(i+1)
            return 0.0
        def p5(preds, truth):
            return sum(1 for p in preds[:5] if p in truth)/5.0

        m = mrr_at_5(ranked, gt[test_q])
        p = p5(ranked, gt[test_q])
        mrrs.append(m)
        p5s.append(p)
        results[test_q] = {'predicted_top5': [(ranked[i], float(probs[idx[i]])) for i in range(min(5,len(ranked)))], 'MRR': m, 'P5': p}

    mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
    mean_p5 = float(np.mean(p5s)) if p5s else 0.0

    # baselines
    def baseline(model_key):
        vals = []
        for q in valid_queries:
            entry = mr['queries'][q]
            lst = []
            if model_key == 'tfidf': lst = [t[0] for t in entry.get('tfidf_only', [])]
            if model_key == 'bm25': lst = [t[0] for t in entry.get('bm25_only', [])]
            if not lst: continue
            for i,pred in enumerate(lst[:5]):
                if pred in gt.get(q, []):
                    vals.append(1.0/(i+1))
                    break
            else:
                vals.append(0.0)
        return float(np.mean(vals)) if vals else 0.0

    baseline_tfidf = baseline('tfidf')
    baseline_bm25 = baseline('bm25')

    out = {'mean_MRR_rich': mean_mrr, 'mean_P5_rich': mean_p5, 'baseline_tfidf': baseline_tfidf, 'baseline_bm25': baseline_bm25, 'per_query': results}
    with open('learn_to_rank_rich_results.json','w',encoding='utf-8') as f:
        json.dump(out,f,indent=2)

    print('Wrote learn_to_rank_rich_results.json')
    print('Mean MRR (rich):', mean_mrr)
    print('Mean P@5 (rich):', mean_p5)
    print('Baseline TF-IDF MRR:', baseline_tfidf)
    print('Baseline BM25 MRR:', baseline_bm25)


if __name__ == '__main__':
    main()
