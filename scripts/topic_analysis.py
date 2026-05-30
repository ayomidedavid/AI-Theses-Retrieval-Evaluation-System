"""
Topic analysis: LDA topic distribution + per-topic model performance heatmap.
Writes:
 - outputs/topic_distribution.png
 - outputs/topic_keywords.csv
 - outputs/per_topic_model_heatmap.png

Usage:
    python scripts/topic_analysis.py --n_topics 10

"""
import os
import re
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    import squarify
except Exception:
    squarify = None

from PyPDF2 import PdfReader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')


def normalize_text(s):
    return re.sub(r'[^a-z0-9]', '', s.lower()) if s else ''


def extract_texts_from_pdfs(base=DATA_DIR, max_pages=10, max_docs=1000):
    texts = []
    filenames = []
    ids = []
    if not os.path.exists(base):
        return ids, filenames, texts
    idx = 0
    for root, dirs, files in os.walk(base):
        for fname in sorted(files):
            if not fname.lower().endswith('.pdf'):
                continue
            path = os.path.join(root, fname)
            try:
                reader = PdfReader(path)
                parts = []
                for i, page in enumerate(reader.pages):
                    if i >= max_pages:
                        break
                    try:
                        txt = page.extract_text() or ''
                    except Exception:
                        txt = ''
                    parts.append(txt)
                full = '\n'.join(parts).strip()
                if not full:
                    continue
                texts.append(full)
                filenames.append(fname)
                ids.append(idx)
                idx += 1
                if max_docs and idx >= max_docs:
                    break
            except Exception as e:
                print('Failed to read', path, e)
                continue
        if max_docs and idx >= max_docs:
            break
    return ids, filenames, texts


def fit_lda(texts, n_topics=10, max_features=10000):
    vect = CountVectorizer(max_df=0.85, min_df=2, max_features=max_features, stop_words='english')
    X = vect.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    W = lda.fit_transform(X)
    return lda, vect, W


def top_keywords(lda, vect, n=12):
    words = np.array(vect.get_feature_names_out())
    topics = []
    for comp in lda.components_:
        idx = comp.argsort()[-n:][::-1]
        topics.append([words[i] for i in idx])
    return topics


def plot_topic_distribution(counts, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    labels = [f'Topic {i}' for i in range(len(counts))]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, counts, color='tab:blue')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Documents')
    ax.set_title('Topic distribution')
    plt.tight_layout()
    path = os.path.join(out_dir, 'topic_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_treemap(counts, out_dir=OUT_DIR):
    if squarify is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    labels = [f'T{idx}\n{c}' for idx, c in enumerate(counts)]
    fig = plt.figure(figsize=(8,6))
    squarify.plot(sizes=counts, label=labels, alpha=0.7)
    plt.axis('off')
    path = os.path.join(out_dir, 'topic_treemap.png')
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def match_title_to_filename(title, filenames):
    # try exact, then normalized substring matching
    if not title:
        return None
    title_n = normalize_text(title)
    for i, fn in enumerate(filenames):
        if normalize_text(fn) == title_n:
            return i
    # substring match
    for i, fn in enumerate(filenames):
        if title_n and title_n in normalize_text(fn):
            return i
    for i, fn in enumerate(filenames):
        if normalize_text(fn) in title_n:
            return i
    return None


def compute_per_topic_metrics(W, filenames, per_query_csv='outputs/per_query_metrics.csv', ground_truth='ground_truth.json', out_dir=OUT_DIR):
    if not os.path.exists(per_query_csv):
        print('Missing per_query_metrics.csv — run compute_and_plot_metrics.py first')
        return None
    df = pd.read_csv(per_query_csv)
    with open(ground_truth, 'r', encoding='utf-8') as fh:
        gt = json.load(fh)

    doc_topic = np.argmax(W, axis=1)
    n_topics = W.shape[1]

    topic_queries = defaultdict(list)
    for q, rels in gt.items():
        assigned = []
        for title in rels:
            midx = match_title_to_filename(title, filenames)
            if midx is not None:
                assigned.append(int(doc_topic[midx]))
        if assigned:
            topic = Counter(assigned).most_common(1)[0][0]
            topic_queries[topic].append(q)

    models = ['tfidf_only', 'bm25_only', 'hybrid_no_optimizer', 'hybrid_with_optimizer']
    heat = np.full((len(models), n_topics), np.nan)
    for t in range(n_topics):
        qs = topic_queries.get(t, [])
        if not qs:
            continue
        rows = df[df['query'].isin(qs)]
        if rows.empty:
            continue
        for mi, m in enumerate(models):
            col = f'{m}_mrr'
            if col in rows.columns:
                heat[mi, t] = float(rows[col].mean())
    # plot heatmap
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, n_topics*0.6), 4))
    im = ax.imshow(np.nan_to_num(heat, nan=0.0), aspect='auto', vmin=0, vmax=1, cmap='magma')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels([f'T{t}' for t in range(n_topics)], rotation=45)
    fig.colorbar(im, ax=ax, label='Mean MRR')
    plt.title('Per-topic MRR across models')
    plt.tight_layout()
    path = os.path.join(out_dir, 'per_topic_model_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    # also write per-topic metrics to CSV for frontend consumption
    rows = []
    for t in range(n_topics):
        for mi, m in enumerate(models):
            val = heat[mi, t]
            rows.append({'topic': int(t), 'model': m, 'mrr': (None if np.isnan(val) else float(val))})
    csv_path = os.path.join(out_dir, 'per_topic_metrics.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return path, topic_queries, heat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_topics', type=int, default=10)
    parser.add_argument('--max_docs', type=int, default=500)
    args = parser.parse_args()

    ids, filenames, texts = extract_texts_from_pdfs(max_docs=args.max_docs)
    if not texts:
        print('No PDF texts found under', DATA_DIR)
        return

    lda, vect, W = fit_lda(texts, n_topics=args.n_topics)
    topics_kw = top_keywords(lda, vect, n=12)

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({'topic': list(range(len(topics_kw))), 'keywords': [', '.join(t) for t in topics_kw]}).to_csv(os.path.join(OUT_DIR, 'topic_keywords.csv'), index=False)

    doc_topic = np.argmax(W, axis=1)
    counts = [int((doc_topic == i).sum()) for i in range(args.n_topics)]
    td = plot_topic_distribution(counts)
    tre = plot_treemap(counts)

    heat_path, topic_queries, heat = compute_per_topic_metrics(W, filenames)
    print('Wrote:', td, tre, heat_path, 'and outputs/topic_keywords.csv')


if __name__ == '__main__':
    main()
