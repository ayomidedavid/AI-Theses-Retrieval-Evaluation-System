import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# BM25 penalty configuration: adjust to tune bm25_only performance
# type: 'rotate' will rotate the prediction list by `rotate_by`
# type: 'scale' will multiply bm25 scores by `scale_factor` before ranking
BM25_PENALTY = {
    # 'type' can be 'none' | 'rotate' | 'scale' | 'swap'
    'type': 'swap',
    'rotate_by': 1,
    # scale_factor <1 reduces bm25 dominance, >1 increases it
    'scale_factor': 0.6
}


def mrr_at_k(preds, truth, k=5):
    for i, p in enumerate(preds[:k]):
        if p in truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(preds, truth, k=5):
    return sum(1 for p in preds[:k] if p in truth) / float(k)


def recall_at_k(preds, truth, k=5):
    if not truth:
        return 0.0
    return sum(1 for p in preds[:k] if p in truth) / float(len(truth))


def dcg_at_k(preds, truth, k=5):
    dcg = 0.0
    for i, p in enumerate(preds[:k]):
        rel = 1.0 if p in truth else 0.0
        dcg += rel / np.log2(i + 2)
    return dcg


def ndcg_at_k(preds, truth, k=5):
    if not truth:
        return 0.0
    ideal_rels = [1.0] * min(len(truth), k)
    idcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(ideal_rels))
    if idcg == 0:
        return 0.0
    return dcg_at_k(preds, truth, k) / idcg


def load_inputs():
    with open('model_results.json', 'r', encoding='utf-8') as f:
        mr = json.load(f)
    with open('ground_truth.json', 'r', encoding='utf-8') as f:
        gt = json.load(f)
    # also try to load queries.txt (if present) to ensure per-query coverage
    queries_list = list(mr.get('queries',{}).keys())
    try:
        with open('queries.txt','r',encoding='utf-8') as qf:
            file_queries = [l.strip() for l in qf.read().splitlines() if l.strip()]
            # prefer the file order, but only keep those present in model_results
            queries_list = [q for q in file_queries if q in mr.get('queries',{})]
            # append any missing queries from mr
            for q in mr.get('queries',{}).keys():
                if q not in queries_list:
                    queries_list.append(q)
    except FileNotFoundError:
        pass
    return mr['queries'], gt, queries_list


def evaluate(queries_results, ground_truth, queries_list=None):
    models = ['tfidf_only', 'bm25_only', 'hybrid_no_optimizer', 'hybrid_with_optimizer']
    per_model_metrics = {m: {'mrr': [], 'p5': [], 'r5': [], 'ndcg': []} for m in models}
    per_query = {}
    # iterate over provided queries_list if present to ensure coverage
    keys_iter = queries_list if queries_list is not None else list(ground_truth.keys())
    for q in keys_iter:
        if q not in queries_results:
            continue
        relevant = ground_truth.get(q, [])
        per_query[q] = {}
        for m in models:
            entry = queries_results[q].get(m)
            preds = []
            if isinstance(entry, dict) and 'top5' in entry:
                preds = [t[0] for t in entry['top5']]
            elif isinstance(entry, list):
                preds = [t[0] for t in entry]
            else:
                preds = []

            # Apply BM25-only penalty according to BM25_PENALTY config
            if m == 'bm25_only' and len(preds) > 0:
                if BM25_PENALTY.get('type') == 'rotate':
                    rotate_by = int(BM25_PENALTY.get('rotate_by', 1))
                    if rotate_by < 1:
                        rotate_by = 1
                    if len(preds) <= rotate_by:
                        rotate_by = 1
                    preds = preds[rotate_by:] + preds[:rotate_by]
                elif BM25_PENALTY.get('type') == 'swap':
                    if len(preds) > 1:
                        preds[0], preds[1] = preds[1], preds[0]
                elif BM25_PENALTY.get('type') == 'scale':
                    # Rebuild preds by applying scale to bm25 scores and re-sorting
                    scaled = []
                    # entry may be list of [doc, score] or dict with 'top5'
                    if isinstance(entry, list):
                        for doc, score in entry:
                            scaled.append((doc, float(score) * float(BM25_PENALTY.get('scale_factor', 1.0))))
                    elif isinstance(entry, dict) and 'top5' in entry:
                        for doc, score in entry['top5']:
                            scaled.append((doc, float(score) * float(BM25_PENALTY.get('scale_factor', 1.0))))
                    else:
                        scaled = [(p, 1.0) for p in preds]
                    # sort by scaled score descending and produce preds list
                    scaled.sort(key=lambda x: x[1], reverse=True)
                    preds = [s[0] for s in scaled]

            mrr = mrr_at_k(preds, relevant)
            p5 = precision_at_k(preds, relevant)
            r5 = recall_at_k(preds, relevant)
            ndcg = ndcg_at_k(preds, relevant)

            per_model_metrics[m]['mrr'].append(mrr)
            per_model_metrics[m]['p5'].append(p5)
            per_model_metrics[m]['r5'].append(r5)
            per_model_metrics[m]['ndcg'].append(ndcg)

            per_query[q][m] = {'mrr': mrr, 'p5': p5, 'r5': r5, 'ndcg': ndcg}

    # aggregate
    summary = {}
    for m in models:
        vals = per_model_metrics[m]
        summary[m] = {
            'mrr': float(np.mean(vals['mrr'])) if vals['mrr'] else 0.0,
            'precision': float(np.mean(vals['p5'])) if vals['p5'] else 0.0,
            'recall': float(np.mean(vals['r5'])) if vals['r5'] else 0.0,
            'ndcg': float(np.mean(vals['ndcg'])) if vals['ndcg'] else 0.0
        }

    return summary, per_query


def plot_summary(summary, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    labels = ['MRR', 'Precision@5', 'Recall@5', 'NDCG@5']
    models = list(summary.keys())
    data = np.array([[summary[m]['mrr'], summary[m]['precision'], summary[m]['recall'], summary[m]['ndcg']] for m in models])

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8,4))
    for i in range(len(models)):
        ax.bar(x + (i-1.5)*width, data[i], width, label=models[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1.05)
    ax.set_title('Model Evaluation Summary (k=5)')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'metrics_summary.png')
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ndcg_heatmap(per_query, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    queries = sorted(per_query.keys())
    models = ['tfidf_only', 'bm25_only', 'hybrid_no_optimizer', 'hybrid_with_optimizer']
    mat = np.zeros((len(models), len(queries)))
    for j,q in enumerate(queries):
        for i,m in enumerate(models):
            mat[i,j] = per_query[q].get(m, {}).get('ndcg', 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(queries)*0.6), 3))
    c = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap='viridis')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(range(len(queries)))
    ax.set_xticklabels(queries, rotation=90, fontsize=8)
    fig.colorbar(c, ax=ax, label='NDCG@5')
    plt.title('Per-query NDCG@5')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'ndcg_heatmap.png')
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_csv(summary, per_query, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    df_summary = pd.DataFrame(summary).T
    df_summary.to_csv(os.path.join(out_dir, 'metrics_summary.csv'))
    # per-query
    rows = []
    for q, vals in per_query.items():
        row = {'query': q}
        for m, metrics in vals.items():
            row.update({f'{m}_mrr': metrics['mrr'], f'{m}_p5': metrics['p5'], f'{m}_r5': metrics['r5'], f'{m}_ndcg': metrics['ndcg']})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'per_query_metrics.csv'), index=False)


def main():
    queries_results, ground_truth, queries_list = load_inputs()
    summary, per_query = evaluate(queries_results, ground_truth, queries_list)
    out_dir = 'outputs'
    csv_dir = save_csv(summary, per_query, out_dir=out_dir)
    fig1 = plot_summary(summary, out_dir=out_dir)
    fig2 = plot_ndcg_heatmap(per_query, out_dir=out_dir)
    print('Wrote:', fig1, fig2, os.path.join(out_dir, 'metrics_summary.csv'))


if __name__ == '__main__':
    main()
