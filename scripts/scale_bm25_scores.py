import json
import sys
import shutil
from datetime import datetime

def usage():
    print('Usage: python scale_bm25_scores.py <scale_factor>')

def main():
    if len(sys.argv) < 2:
        usage(); sys.exit(2)
    try:
        factor = float(sys.argv[1])
    except Exception:
        usage(); sys.exit(2)

    src = 'model_results.json'
    bak = f'model_results.json.bak.{datetime.now().strftime("%Y%m%d%H%M%S")}'
    shutil.copyfile(src, bak)
    print('Backup saved to', bak)

    with open(src, 'r', encoding='utf-8') as f:
        mr = json.load(f)

    for q, entry in mr.get('queries', {}).items():
        bm = entry.get('bm25_only')
        if isinstance(bm, list):
            newbm = []
            for item in bm:
                if isinstance(item, list) and len(item) >= 2:
                    doc = item[0]
                    try:
                        score = float(item[1])
                    except Exception:
                        score = 0.0
                    newbm.append([doc, score * factor])
                else:
                    newbm.append(item)
            mr['queries'][q]['bm25_only'] = newbm

    with open(src, 'w', encoding='utf-8') as f:
        json.dump(mr, f, indent=2)
    print('Scaled bm25_only scores by', factor)

if __name__ == '__main__':
    main()
