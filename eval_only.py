import json, csv
from pathlib import Path

def MRR(preds,truth):
    for i,p in enumerate(preds):
        if p in truth:
            return 1.0/(i+1)
    return 0.0

def precision_at_k(preds, truth, k=5):
    if len(preds)==0: return 0.0
    return sum(1 for p in preds[:k] if p in truth)/float(k)

def recall_at_k(preds, truth, k=5):
    if len(truth)==0: return None
    return sum(1 for p in preds[:k] if p in truth)/float(len(truth))

mr_path = Path('model_results.json')
if not mr_path.exists():
    print('model_results.json not found; run full notebook first')
    raise SystemExit(2)
results = json.loads(mr_path.read_text(encoding='utf-8'))['queries']

# load queries
q_path = Path('queries.txt')
if q_path.exists():
    queries = [q.strip() for q in q_path.read_text(encoding='utf-8').splitlines() if q.strip()]
else:
    queries = list(results.keys())

# load ground truth
gt_path = Path('ground_truth.json')
if gt_path.exists():
    gt = json.loads(gt_path.read_text(encoding='utf-8'))
else:
    gt = {q: [] for q in queries}
    gt_path.write_text(json.dumps(gt, indent=2), encoding='utf-8')
    print('Wrote ground_truth.json template; please fill it if you want meaningful evaluation')

eval_out = {}
rows = []
for q in queries:
    preds_tfidf = [fn for fn,_ in results.get(q, {}).get('tfidf_only', [])]
    preds_bm25 = [fn for fn,_ in results.get(q, {}).get('bm25_only', [])]
    preds_hybrid_opt = [fn for fn,_ in results.get(q, {}).get('hybrid_with_optimizer', {}).get('top5', [])]
    preds_hybrid_fixed = [fn for fn,_ in results.get(q, {}).get('hybrid_no_optimizer', {}).get('top5', [])]
    truth = set(gt.get(q, []))
    m_tfidf = MRR(preds_tfidf, truth)
    p5_tfidf = precision_at_k(preds_tfidf, truth, 5)
    r5_tfidf = recall_at_k(preds_tfidf, truth, 5)
    m_bm25 = MRR(preds_bm25, truth)
    p5_bm25 = precision_at_k(preds_bm25, truth, 5)
    r5_bm25 = recall_at_k(preds_bm25, truth, 5)
    m_hopt = MRR(preds_hybrid_opt, truth)
    p5_hopt = precision_at_k(preds_hybrid_opt, truth, 5)
    r5_hopt = recall_at_k(preds_hybrid_opt, truth, 5)
    m_hfix = MRR(preds_hybrid_fixed, truth)
    p5_hfix = precision_at_k(preds_hybrid_fixed, truth, 5)
    r5_hfix = recall_at_k(preds_hybrid_fixed, truth, 5)
    eval_out[q] = {
        'MRR_tfidf': m_tfidf, 'P5_tfidf': p5_tfidf, 'R5_tfidf': r5_tfidf,
        'MRR_bm25': m_bm25, 'P5_bm25': p5_bm25, 'R5_bm25': r5_bm25,
        'MRR_hybrid_opt': m_hopt, 'P5_hybrid_opt': p5_hopt, 'R5_hybrid_opt': r5_hopt,
        'MRR_hybrid_fixed': m_hfix, 'P5_hybrid_fixed': p5_hfix, 'R5_hybrid_fixed': r5_hfix,
    }
    rows.append([q, m_tfidf, p5_tfidf, r5_tfidf, m_bm25, p5_bm25, r5_bm25, m_hopt, p5_hopt, r5_hopt, m_hfix, p5_hfix, r5_hfix])

Path('model_results_eval.json').write_text(json.dumps(eval_out, indent=2), encoding='utf-8')
with open('model_results_eval.csv','w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['query','MRR_tfidf','P5_tfidf','R5_tfidf','MRR_bm25','P5_bm25','R5_bm25','MRR_hybrid_opt','P5_hybrid_opt','R5_hybrid_opt','MRR_hybrid_fixed','P5_hybrid_fixed','R5_hybrid_fixed'])
    writer.writerows(rows)
print('Wrote model_results_eval.json and model_results_eval.csv')
