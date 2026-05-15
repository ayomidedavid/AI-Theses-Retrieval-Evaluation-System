const API_BASE = "/api";

document.addEventListener("DOMContentLoaded", () => {
    // Theme initialization: read stored preference and apply
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    const applyTheme = (theme) => {
        if (theme === 'light') {
            document.body.classList.add('light-theme');
            if (themeIcon) themeIcon.textContent = '🌞';
            if (themeToggle) themeToggle.textContent = '🌞';
        } else {
            document.body.classList.remove('light-theme');
            if (themeIcon) themeIcon.textContent = '🌙';
            if (themeToggle) themeToggle.textContent = '🌙';
        }
        try { localStorage.setItem('theme', theme); } catch(e){}
    };
    const stored = (() => { try { return localStorage.getItem('theme'); } catch(e){ return null; } })();
    applyTheme(stored === 'light' ? 'light' : 'dark');

    if (themeToggle) themeToggle.addEventListener('click', () => {
        const isLight = document.body.classList.contains('light-theme');
        applyTheme(isLight ? 'dark' : 'light');
    });
    const searchInput = document.getElementById("searchInput");
    const searchBtn = document.getElementById("searchBtn");
    const resultsContainer = document.getElementById("resultsContainer");
    const ingestLocalBtn = document.getElementById("ingestLocalBtn");
    const evaluateBtn = document.getElementById("evaluateBtn");
    const statusMessage = document.getElementById("statusMessage");
    const logoutBtn = document.getElementById("logoutBtn");

    // Fetch and display user name; hide admin controls for non-admins
    fetch('/api/me', { credentials: 'include' })
        .then(r => r.json())
        .then(data => {
            if (data.name) {
                document.getElementById('userGreeting').textContent = `Welcome, ${data.name}`;
                // Only show admin controls for admin users
                if (data.role !== 'admin') {
                    ingestLocalBtn.style.display = 'none';
                    evaluateBtn.style.display = 'none';
                }

                // Fetch years for the dropdown filters
                fetch('/api/years', { credentials: 'include' })
                    .then(res => res.json())
                    .then(yearData => {
                        if (yearData.years && yearData.years.length > 0) {
                            const startSelect = document.getElementById('startYear');
                            const endSelect = document.getElementById('endYear');
                            yearData.years.forEach(year => {
                                const opt1 = document.createElement('option');
                                opt1.value = year;
                                opt1.textContent = year;
                                startSelect.appendChild(opt1);

                                const opt2 = document.createElement('option');
                                opt2.value = year;
                                opt2.textContent = year;
                                endSelect.appendChild(opt2);
                            });
                        }
                    })
                    .catch(e => console.error("Could not load years", e));
            } else {
                window.location.href = '/login';
            }
        })
        .catch(() => window.location.href = '/login');

    // Logout
    logoutBtn.addEventListener('click', async () => {
        await fetch('/api/logout', { method: 'POST', credentials: 'include' });
        window.location.href = '/login';
    });

    // Utilities
    const showMessage = (msg, isError = false) => {
        statusMessage.textContent = msg;
        statusMessage.style.color = isError ? "#ef4444" : "#3b82f6";
        statusMessage.classList.add("show");
        setTimeout(() => statusMessage.classList.remove("show"), 5000);
    };

    const renderResults = (results) => {
        resultsContainer.innerHTML = "";
        
        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="initial-state">
                    No matching theses found. Try different keywords.
                </div>
            `;
            return;
        }

        results.forEach(res => {
            const card = document.createElement("div");
            card.className = "result-card";
            
            // Limit abstract length gracefully if its too long
            const abstract = res.abstractSnippet ? res.abstractSnippet.substring(0, 300) + '...' : "No abstract available.";
            const yearBadge = res.year ? `<span class="metric-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">Year: ${res.year}</span>` : '';
            const valueBadge = typeof res.valueScore === 'number'
                ? `<span class="metric-badge" title="Ensemble Value Score">Value: ${res.valueScore.toFixed(4)}</span>`
                : '';

            card.innerHTML = `
                <h3>${res.title}</h3>
                <div class="metrics">
                    <span class="metric-badge" title="Combined Ensemble Score">Ensemble: ${(res.score * 100).toFixed(1)}%</span>
                    ${valueBadge}
                    ${yearBadge}
                    <span class="metric-badge" title="Term Frequency Bias">TF-IDF: ${res.tfidf.toFixed(3)}</span>
                    <span class="metric-badge" title="Length Normalised Bias">BM25: ${res.bm25.toFixed(2)}</span>
                </div>
                <p>${abstract}</p>
            `;
            resultsContainer.appendChild(card);
        });
    };

    // Actions
    const performSearch = async () => {
        const query = searchInput.value.trim();
        const startYear = document.getElementById("startYear") ? document.getElementById("startYear").value.trim() : "";
        const endYear = document.getElementById("endYear") ? document.getElementById("endYear").value.trim() : "";

        if (!query) {
            showMessage("Please enter a search query.", true);
            return;
        }

        resultsContainer.innerHTML = '<div class="initial-state">Searching...</div>';
        
        let searchUrl = `${API_BASE}/search?q=${encodeURIComponent(query)}`;
        if (startYear) searchUrl += `&startYear=${startYear}`;
        if (endYear) searchUrl += `&endYear=${endYear}`;
        
        try {
            const res = await fetch(searchUrl);
            const data = await res.json();
            
            if (res.ok) {
                renderResults(data.results);
            } else {
                showMessage(data.error || "Search failed", true);
                resultsContainer.innerHTML = `<div class="initial-state">${data.error || "Search failed"}</div>`;
            }
        } catch (err) {
            console.error(err);
            showMessage("Network Error while searching.", true);
            resultsContainer.innerHTML = '<div class="initial-state" style="color:#ef4444">API Connection failed.</div>';
        }
    };

    const ingestLocalFiles = async () => {
        showMessage("Ingesting local files... This might take a few minutes depending on PDF length.");
        ingestLocalBtn.disabled = true;
        
        try {
            const res = await fetch(`${API_BASE}/ingest_local`, { method: "POST" });
            const data = await res.json();
            
            if (res.ok) {
                showMessage(data.message);
            } else {
                showMessage(data.error || "Ingestion failed.", true);
            }
        } catch (err) {
            showMessage("Network error during ingestion.", true);
        } finally {
            ingestLocalBtn.disabled = false;
        }
    };

    const evaluateSystem = async () => {
        showMessage("Running Evaluation Metrics against Ground Truth...");
        evaluateBtn.disabled = true;
        resultsContainer.innerHTML = '<div class="initial-state">Calculating MRR, Precision, and Recall...</div>';
        
        try {
            const res = await fetch(`${API_BASE}/evaluate`, { method: "GET" });
            const data = await res.json();
            
            if (res.ok) {
                showMessage("Evaluation Complete.");
                resultsContainer.innerHTML = `
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                            <h3 style="margin: 0;">System Evaluation Metrics</h3>
                            <span class="metric-badge" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3);">
                                Total Articles: ${data.total_articles}
                            </span>
                        </div>
                        <p style="margin-bottom: 1.5rem; color: var(--text-muted); font-size: 0.9rem;">Averages across Ground-truth queries (k=5)</p>
                        
                        <div class="chart-container" style="position: relative; height:300px; width:100%; margin-bottom: 2rem;">
                            <canvas id="metricsChart"></canvas>
                        </div>

                        <table style="width: 100%; border-collapse: collapse; color: var(--text-main); margin-top: 1rem;">
                            <tr style="border-bottom: 1px solid var(--border-color); text-align: left;">
                                <th style="padding: 10px;">Model</th>
                                <th style="padding: 10px;">MRR</th>
                                <th style="padding: 10px;">Precision@5</th>
                                <th style="padding: 10px;">Recall@5</th>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 10px; color: var(--accent);">BMTS Ensemble</td>
                                <td style="padding: 10px;">${data.ensemble.mrr.toFixed(3)}</td>
                                <td style="padding: 10px;">${data.ensemble.precision.toFixed(3)}</td>
                                <td style="padding: 10px;">${data.ensemble.recall.toFixed(3)}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 10px; color: var(--text-muted);">TF-IDF Only</td>
                                <td style="padding: 10px;">${data.tf_idf.mrr.toFixed(3)}</td>
                                <td style="padding: 10px;">${data.tf_idf.precision.toFixed(3)}</td>
                                <td style="padding: 10px;">${data.tf_idf.recall.toFixed(3)}</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px; color: var(--text-muted);">BM25 Only</td>
                                <td style="padding: 10px;">${data.bm25.mrr.toFixed(3)}</td>
                                <td style="padding: 10px;">${data.bm25.precision.toFixed(3)}</td>
                                <td style="padding: 10px;">${data.bm25.recall.toFixed(3)}</td>
                            </tr>
                        </table>
                    </div>
                `;

                // Render Chart
                const ctx = document.getElementById('metricsChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['MRR', 'Precision@5', 'Recall@5'],
                        datasets: [
                            {
                                label: 'BMTS Ensemble',
                                data: [data.ensemble.mrr, data.ensemble.precision, data.ensemble.recall],
                                backgroundColor: 'rgba(139, 92, 246, 0.6)',
                                borderColor: 'rgba(139, 92, 246, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'TF-IDF',
                                data: [data.tf_idf.mrr, data.tf_idf.precision, data.tf_idf.recall],
                                backgroundColor: 'rgba(59, 130, 246, 0.4)',
                                borderColor: 'rgba(59, 130, 246, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'BM25',
                                data: [data.bm25.mrr, data.bm25.precision, data.bm25.recall],
                                backgroundColor: 'rgba(148, 163, 184, 0.3)',
                                borderColor: 'rgba(148, 163, 184, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1.1,
                                grid: { color: 'rgba(148, 163, 184, 0.1)' },
                                ticks: { color: '#94a3b8' }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: '#94a3b8' }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#f8fafc', font: { family: 'Inter' } }
                            }
                        }
                    }
                });
            } else {
                showMessage(data.error || "Evaluation failed.", true);
                resultsContainer.innerHTML = `<div class="initial-state">${data.error}</div>`;
            }
        } catch (err) {
            showMessage("Network error during evaluation.", true);
            resultsContainer.innerHTML = `<div class="initial-state" style="color:red">Failed to reach API.</div`;
        } finally {
            evaluateBtn.disabled = false;
        }
    };

    // Listeners
    searchBtn.addEventListener("click", performSearch);
    searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") performSearch();
    });
    ingestLocalBtn.addEventListener("click", ingestLocalFiles);
    evaluateBtn.addEventListener("click", evaluateSystem);
});
