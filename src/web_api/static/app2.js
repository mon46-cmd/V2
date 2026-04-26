/* app2.js — markers, logs, portfolio extensions */
'use strict';

// Extend state
S.markers = []; S.showFlagMarkers = true; S.showAiMarkers = true;
S.logFilter = ''; S.logTimer = null;

// ── Marker toggles ────────────────────────────────────────────────────────
document.querySelectorAll('.mtog').forEach(b => b.addEventListener('click', () => {
  b.classList.toggle('active');
  if (b.dataset.mtog === 'flags') S.showFlagMarkers = b.classList.contains('active');
  if (b.dataset.mtog === 'ai')    S.showAiMarkers   = b.classList.contains('active');
  applyMarkers();
}));

function applyMarkers() {
  if (!S.cSeries) return;
  const visible = S.markers.filter(m =>
    (m.type === 'flag'    && S.showFlagMarkers) ||
    (m.type === 'ai_call' && S.showAiMarkers)
  );
  S.cSeries.setMarkers(visible);
}

async function loadMarkers() {
  const d = await get(`/api/markers/${S.sym}?tf=${S.tf}`);
  if (!d) return;
  S.markers = d.markers || [];
  applyMarkers();
}

// ── Patch goPage to handle new pages ─────────────────────────────────────
const _origGoPage = goPage;
window.goPage = function(page) {
  _origGoPage(page);
  if (page === 'logs')      { startLogs(); }
  if (page === 'portfolio') { loadPortfolio(); }
};

// ── Patch loadChartData to also load markers ──────────────────────────────
const _origLoadChart = loadChartData;
window.loadChartData = async function() {
  await _origLoadChart();
  loadMarkers();
};

// ── Logs ──────────────────────────────────────────────────────────────────
async function fetchLogs() {
  const box = document.getElementById('log-box');
  if (!box) return;
  const lvl = S.logFilter ? `&level=${S.logFilter}` : '';
  const d = await get(`/api/logs?n=400${lvl}`);
  if (!d) return;
  const auto = document.getElementById('log-auto')?.checked;
  box.innerHTML = d.lines.length ? d.lines.map(l => {
    const msg = (l.msg||'').replace(/</g,'&lt;');
    return `<div class="log-line"><span class="log-ts">${l.ts_h||''}</span><span class="log-lvl ${l.level||''}">${l.level||''}</span><span class="log-name">${(l.name||'').split('.').pop()}</span><span class="log-msg">${msg}</span></div>`;
  }).join('') : '<div class="log-empty">No log entries yet</div>';
  if (auto) box.scrollTop = box.scrollHeight;
}

function startLogs() {
  fetchLogs();
  if (S.logTimer) clearInterval(S.logTimer);
  S.logTimer = setInterval(() => { if (S.page === 'logs') fetchLogs(); }, 3000);
}

document.querySelectorAll('.tab[data-lf]').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.tab[data-lf]').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  S.logFilter = b.dataset.lf;
  fetchLogs();
}));
document.getElementById('log-refresh')?.addEventListener('click', fetchLogs);

// ── Portfolio ─────────────────────────────────────────────────────────────
async function loadPortfolio() {
  const kpis = document.getElementById('port-kpis');
  const syms = document.getElementById('port-symbols');
  const curve = document.getElementById('port-curve-box');
  if (!kpis) return;
  kpis.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  const d = await get('/api/portfolio');
  if (!d) { kpis.innerHTML = '<div class="empty">No data</div>'; return; }

  const pnl = d.equity_current - d.equity_start;
  const pnlPct = (pnl / d.equity_start * 100).toFixed(2);

  kpis.innerHTML = [
    ['Starting Equity', `$${d.equity_start.toLocaleString()}`, ''],
    ['Current Equity',  `$${d.equity_current.toFixed(4)}`, pnl >= 0 ? 'gr' : 'rd'],
    ['P&L',  `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(4)} (${pnlPct}%)`, pnl >= 0 ? 'gr' : 'rd'],
    ['AI Cost Total', `$${d.total_ai_cost.toFixed(5)}`, 'gd'],
    ['Total AI Calls', d.calls_total, ''],
    ['Watched Symbols', (d.watched_symbols||[]).length, ''],
  ].map(([l,v,c]) => `<div class="scard"><div class="scard-lbl">${l}</div><div class="scard-val ${c}">${v}</div></div>`).join('');

  // Watched symbols table
  if (syms) syms.innerHTML = (d.watched_symbols||[]).map(s => {
    const act = s.last_action || 'null';
    const ts = s.last_ts ? s.last_ts.slice(0,16) : '—';
    return `<div class="sym-row"><span class="sym-row-sym">${s.symbol}</span><span style="color:var(--tx3);font-size:.72rem">${s.analyses}A / ${s.reviews}R · ${ts}</span><span class="sym-row-action ${act}">${act}</span></div>`;
  }).join('') || '<div class="log-empty">No symbols tracked yet</div>';

  // Mini equity sparkline
  if (curve && d.equity_curve?.length > 1) {
    const pts = d.equity_curve.map(p => p.equity);
    const mn = Math.min(...pts), mx = Math.max(...pts), rng = mx - mn || 1;
    const w = curve.clientWidth || 300, h = 180;
    const step = w / (pts.length - 1);
    const path = pts.map((v, i) => `${i * step},${h - ((v - mn) / rng) * (h - 20) - 10}`).join(' ');
    const up = pts[pts.length-1] >= pts[0];
    const col = up ? '#1db88c' : '#e85555';
    curve.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="overflow:visible">
      <polyline points="${path}" fill="none" stroke="${col}" stroke-width="2" opacity=".9"/>
      <circle cx="${(pts.length-1)*step}" cy="${h-((pts[pts.length-1]-mn)/rng)*(h-20)-10}" r="4" fill="${col}"/>
    </svg>`;
  } else if (curve) {
    curve.innerHTML = '<div class="log-empty">Not enough data for equity curve</div>';
  }
}

// ── Wire nav for new pages (patch existing goPage) ───────────────────────
// app.js already calls goPage on nav clicks; we've patched window.goPage above,
// so new pages (logs, portfolio) are handled automatically.
