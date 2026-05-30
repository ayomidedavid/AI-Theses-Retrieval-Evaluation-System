// Frontend is a static page. Load metrics directly from ../outputs/ by default.
const OUTPUTS_BASE = '../outputs';

function parseCSV(text) {
  const rows = text.trim().split('\n').map(r => r.split(','));
  const header = rows[0].map(h => h.trim());
  const data = rows.slice(1).map(r => r.map(c => c.trim()));
  return { header, data };
}

function createCanvasCard(title) {
  const wrapper = document.createElement('div');
  wrapper.className = 'chart-card';
  wrapper.style.width = '640px';
  wrapper.style.marginBottom = '1rem';
  const h = document.createElement('h3'); h.textContent = title; wrapper.appendChild(h);
  const canvas = document.createElement('canvas'); wrapper.appendChild(canvas);
  return { wrapper, canvas };
}

function makeBarChart(ctx, labels, datasets, opts={}){
  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: Object.assign({ responsive:true, maintainAspectRatio:false }, opts)
  });
}

async function render() {
  const root = document.getElementById('chartsRoot');
  const imgsRoot = document.getElementById('imagesRoot');
  try {
    // Show available plot images if present
    const imgs = ['metrics_summary.png','ndcg_heatmap.png'];
    imgs.forEach(name => {
      const url = `${OUTPUTS_BASE}/${name}`;
      const card = document.createElement('div');
      card.style.width = '260px'; card.style.background='var(--card-bg)'; card.style.padding='0.5rem'; card.style.borderRadius='6px';
      const img = document.createElement('img'); img.src = url; img.style.width='100%'; img.alt = name;
      const lbl = document.createElement('div'); lbl.textContent = name; lbl.style.fontSize='0.8rem'; lbl.style.marginTop='0.35rem';
      card.appendChild(img); card.appendChild(lbl); imgsRoot.appendChild(card);
    });

    // Load metrics summary CSV
    const sumResp = await fetch(`${OUTPUTS_BASE}/metrics_summary.csv`);
    if (sumResp.ok) {
      const txt = await sumResp.text();
      const { header, data } = parseCSV(txt);
      // compute averages across rows (usually single row)
      const nums = {};
      header.forEach(h=> nums[h]=[]);
      data.forEach(row => row.forEach((cell,ci)=>{ const v=parseFloat(cell); if(!isNaN(v)) nums[header[ci]].push(v); }));
      const avg = header.map(h=> { const arr=nums[h]; return arr && arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length) : 0; });
      const { wrapper, canvas } = createCanvasCard('Metrics Summary'); root.appendChild(wrapper);
      const ctx = canvas.getContext('2d');
      const labels = header.slice(1); // skip empty first column if present
      const values = avg.slice(1);
      makeBarChart(ctx, labels, [{ label: 'Average', data: values, backgroundColor:'rgba(59,130,246,0.7)'}], { scales:{ y:{ beginAtZero:true, suggestedMax:1 } } });
    }

    // Load per-query metrics and render NDGC table + simple per-query MRR chart
    const perResp = await fetch(`${OUTPUTS_BASE}/per_query_metrics.csv`);
    if (perResp.ok) {
      const txt = await perResp.text();
      const { header, data } = parseCSV(txt);
      const low = header.map(h=>h.toLowerCase());
      const queries = data.map(r=> r[0]);
      // gather ndcg columns for each model
      const models = ['tfidf_only','bm25_only','hybrid_no_optimizer','hybrid_with_optimizer'];
      const ndcgMat = models.map(m => data.map(r => parseFloat(r[low.indexOf(m + '_ndcg')]) || 0));

      // create table heatmap
      const tbl = document.createElement('table'); tbl.style.borderCollapse='collapse'; tbl.style.marginTop='0.5rem';
      const thead = document.createElement('thead'); const trh = document.createElement('tr'); const thEmpty = document.createElement('th'); thEmpty.textContent=''; trh.appendChild(thEmpty);
      models.forEach(m=>{ const th=document.createElement('th'); th.textContent=m; th.style.padding='6px'; trh.appendChild(th); }); thead.appendChild(trh); tbl.appendChild(thead);
      const tbody = document.createElement('tbody');
      queries.forEach((q,i)=>{
        const tr = document.createElement('tr'); const tdq = document.createElement('td'); tdq.textContent = q; tdq.style.padding='6px'; tr.appendChild(tdq);
        models.forEach((m,mi)=>{ const td = document.createElement('td'); const v = ndcgMat[mi][i] || 0; td.textContent = v.toFixed(3); td.style.padding='6px'; td.style.background = `rgba(59,130,246,${v*0.9+0.05})`; tr.appendChild(td); });
        tbody.appendChild(tr);
      }); tbl.appendChild(tbody);
      const box = document.createElement('div'); box.style.overflow='auto'; box.style.marginBottom='1rem'; box.appendChild(tbl);
      const hdr = document.createElement('h3'); hdr.textContent = 'Per-query NDCG Heatmap'; root.appendChild(hdr); root.appendChild(box);

      // Simple per-query MRR bar chart for models (averaged)
      const avgMrrs = models.map((m,mi) => { const vals = data.map(r=> parseFloat(r[low.indexOf(m + '_mrr')]) || 0); return vals.reduce((a,b)=>a+b,0)/vals.length; });
      const { wrapper: wM, canvas: cM } = createCanvasCard('Average Per-Model MRR'); root.appendChild(wM);
      makeBarChart(cM.getContext('2d'), models, [{ label: 'MRR', data: avgMrrs, backgroundColor:'rgba(139,92,246,0.7)'}], { scales:{ y:{ beginAtZero:true, suggestedMax:1 } } });
    }

  } catch (e) {
    console.error('Could not load charts', e);
    document.getElementById('chartsRoot').textContent = 'Could not load charts.';
  }
}

window.addEventListener('DOMContentLoaded', render);
