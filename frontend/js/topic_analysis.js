const API_BASE = '/api';
const PUBLIC_BASE = '/public';

async function loadMetrics() {
    try {
        // prefer public metrics endpoint so non-admin users can view charts
        const res = await fetch(`${PUBLIC_BASE}/metrics_list`);
        if (!res.ok) {
            document.getElementById('galleryInner').textContent = 'No metrics available.';
            return;
        }
        const data = await res.json();
        const files = data.files || [];

        // Find specific files
        const keywords = files.find(f => f.toLowerCase().includes('topic_keywords'));
        const perTopic = files.find(f => f.toLowerCase().includes('per_topic_metrics')) || files.find(f => f.toLowerCase().includes('per_topic'));
        const dist = files.find(f => f.toLowerCase().includes('topic_distribution'));
        const heat = files.find(f => f.toLowerCase().includes('per_topic_model_heatmap')) || files.find(f => f.toLowerCase().includes('per_topic'));

        if (keywords) {
            document.getElementById('downloadKeywords').style.display = 'inline-block';
            document.getElementById('downloadKeywords').href = `${PUBLIC_BASE}/metrics/${encodeURIComponent(keywords)}`;
            const csv = await fetch(`${PUBLIC_BASE}/metrics/${encodeURIComponent(keywords)}`);
            const txt = await csv.text();
            const rows = txt.trim().split('\n').map(r => r.split(','));
            const table = document.createElement('table');
            table.style.width = '100%';
            const thead = document.createElement('thead');
            const trh = document.createElement('tr');
            rows[0].forEach(h => { const th = document.createElement('th'); th.textContent = h; trh.appendChild(th); });
            thead.appendChild(trh);
            table.appendChild(thead);
            const tb = document.createElement('tbody');
            rows.slice(1).forEach(r => { const tr = document.createElement('tr'); r.forEach(c => { const td = document.createElement('td'); td.textContent = c; tr.appendChild(td); }); tb.appendChild(tr); });
            table.appendChild(tb);
            const div = document.getElementById('keywordsTable');
            div.innerHTML = '';
            div.appendChild(table);
        }

        if (dist) {
            document.getElementById('distributionImg').src = `${PUBLIC_BASE}/metrics/${encodeURIComponent(dist)}`;
        }
        if (heat) {
            document.getElementById('heatmapImg').src = `${PUBLIC_BASE}/metrics/${encodeURIComponent(heat)}`;
        }

        if (perTopic) {
            document.getElementById('downloadPerTopic').style.display = 'inline-block';
            document.getElementById('downloadPerTopic').href = `${PUBLIC_BASE}/metrics/${encodeURIComponent(perTopic)}`;
            const csv = await fetch(`${PUBLIC_BASE}/metrics/${encodeURIComponent(perTopic)}`);
            const txt = await csv.text();
            const rows = txt.trim().split('\n').map(r => r.split(','));
            const table = document.createElement('table');
            table.style.width = '100%';
            const thead = document.createElement('thead');
            const trh = document.createElement('tr');
            rows[0].forEach(h => { const th = document.createElement('th'); th.textContent = h; trh.appendChild(th); });
            thead.appendChild(trh);
            table.appendChild(thead);
            const tb = document.createElement('tbody');
            rows.slice(1).forEach(r => { const tr = document.createElement('tr'); r.forEach(c => { const td = document.createElement('td'); td.textContent = c; tr.appendChild(td); }); tb.appendChild(tr); });
            table.appendChild(tb);
            const div = document.getElementById('perTopicTable');
            div.innerHTML = '';
            div.appendChild(table);
        }

        // gallery of all images
        const gallery = document.getElementById('galleryInner');
        gallery.innerHTML = '';
        files.filter(f => f.toLowerCase().endsWith('.png') || f.toLowerCase().endsWith('.jpg')).forEach(name => {
            const wrapper = document.createElement('div');
            wrapper.style.width = '240px';
            wrapper.style.background = 'var(--card-bg)';
            wrapper.style.padding = '0.5rem';
            wrapper.style.borderRadius = '6px';
            const img = document.createElement('img');
            img.src = `${PUBLIC_BASE}/metrics/${encodeURIComponent(name)}`;
            img.alt = name;
            img.style.width = '100%';
            img.style.height = 'auto';
            wrapper.appendChild(img);
            const lbl = document.createElement('div'); lbl.textContent = name; lbl.style.fontSize = '0.8rem'; lbl.style.marginTop = '0.35rem';
            wrapper.appendChild(lbl);
            gallery.appendChild(wrapper);
        });

    } catch (e) {
        console.error('Could not load metrics', e);
        document.getElementById('galleryInner').textContent = 'Could not load metrics.';
    }
}

window.addEventListener('DOMContentLoaded', loadMetrics);
