import json, sys
from pathlib import Path
nb_path = Path('model.ipynb')
if not nb_path.exists():
    print('model.ipynb not found')
    sys.exit(2)
nb = json.loads(nb_path.read_text(encoding='utf-8'))
# Execute code cells in order
env = {}
for cell in nb.get('cells',[]):
    if cell.get('cell_type')!='code':
        continue
    src = ''.join(cell.get('source',[]))
    print('\n--- Executing cell ---\n')
    try:
        exec(src, env)
    except SystemExit as e:
        print('Cell raised SystemExit:', e)
    except Exception as e:
        print('Error executing cell:', e)
print('\nDone running notebook cells')
