/* Trading Dashboard */
'use strict';

// ── State ────────────────────────────────────────────────────────────────────
const S = {
  page: 'market', sym: 'BTCUSDT', tf: '15m',
  universe: [], aiPicks: new Set(),
  chart: null, cSeries: null, vSeries: null, cRO: null,
  callsPage: 1, callsFilter: '',
  sparkLoaded: new Set(), sparkObs: null,
};

// ── Helpers ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const qsa = sel => [...document.querySelectorAll(sel)];
const nf = () => new Promise(r => requestAnimationFrame(r));
const esc = s => { const d = document.createElement('div'); d.textContent = s||''; return d.innerHTML; };

async function get(path) {
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(r.status);
    return r.json();
  } catch(e) { console.error('[API]', path, e.message); return null; }
}

function fP(p) {
  if (p == null) return '—';
  if (p >= 10000) return p.toLocaleString('en',{minimumFractionDigits:0,maximumFractionDigits:0});
  if (p >= 1000) return p.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});
  if (p >= 1) return p.toFixed(4);
  if (p >= 0.001) return p.toFixed(6);
  return p.toPrecision(4);
}
function fPct(p) { if(p==null)return'—'; return (p>=0?'+':'')+(p*100).toFixed(2)+'%'; }
function fVol(v) {
  if(v==null)return'—';
  if(v>=1e9)return(v/1e9).toFixed(1)+'B';
  if(v>=1e6)return(v/1e6).toFixed(1)+'M';
  if(v>=1e3)return(v/1e3).toFixed(0)+'K';
  return v.toFixed(0);
}
function fT(iso) {
  if(!iso)return'—';
  return new Date(iso).toLocaleString('en-GB',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
}

// ── Navigation ───────────────────────────────────────────────────────────────
const TITLES = {market:'Market Overview', chart:'Chart', calls:'AI Call Log', status:'Status'};

function goPage(page) {
  qsa('.pg').forEach(p => p.classList.remove('active'));
  qsa('.nb').forEach(b => b.classList.remove('active'));
  const pg = $(`pg-${page}`), btn = document.querySelector(`.nb[data-page="${page}"]`);
  if (pg) pg.classList.add('active');
  if (btn) btn.classList.add('active');
  $('ptitle').textContent = TITLES[page] || page;
  S.page = page;
  if (page === 'market') showMarket();
  if (page === 'chart')  showChart();
  if (page === 'calls')  loadCalls();
  if (page === 'status') loadStatus();
}

qsa('.nb[data-page]').forEach(b => b.addEventListener('click', () => window.goPage(b.dataset.page)));

// ── Market ───────────────────────────────────────────────────────────────────
async function fetchUniverse() {
  const [wl, uni] = await Promise.all([get('/api/watchlist'), get('/api/universe')]);
  S.aiPicks.clear();
  (wl?.picks || []).forEach(p => S.aiPicks.add(p.symbol || ''));
  S.universe = uni?.symbols || [];
  console.log('[Market] universe count=', S.universe.length);
}

async function showMarket() {
  if (!S.universe.length) {
    renderSkeletons();
    await fetchUniverse();
  }
  renderMarket();
}

function renderSkeletons() {
  $('pgrid').innerHTML = Array(12).fill('<div class="pcard skel" style="height:118px"></div>').join('');
}

function renderMarket() {
  const grid = $('pgrid');
  if (!grid) return;
  const mf = document.querySelector('.tab.active[data-mf]')?.dataset.mf || 'all';
  const q = ($('msearch')?.value || '').toLowerCase();
  let list = [...S.universe];
  if (q) list = list.filter(s => s.symbol.toLowerCase().includes(q));
  if (mf === 'ai') list = list.filter(s => S.aiPicks.has(s.symbol));
  else if (mf === 'up') list = list.filter(s => (s.change_24h||0) > 0).sort((a,b) => (b.change_24h||0)-(a.change_24h||0));
  else if (mf === 'dn') list = list.filter(s => (s.change_24h||0) < 0).sort((a,b) => (a.change_24h||0)-(b.change_24h||0));
  else list.sort((a,b) => {
    const ai = S.aiPicks.has(a.symbol)?1:0, bi = S.aiPicks.has(b.symbol)?1:0;
    if (ai!==bi) return bi-ai;
    return (b.turnover_24h||0)-(a.turnover_24h||0);
  });

  const mcnt = $('mcnt');
  if (mcnt) mcnt.textContent = list.length;

  if (!list.length) { grid.innerHTML = '<div class="empty">No pairs match filter</div>'; return; }

  grid.innerHTML = list.map(s => {
    const ai = S.aiPicks.has(s.symbol);
    const chg = s.change_24h, cc = chg!=null?(chg>=0?'up':'dn'):'';
    const base = s.symbol.replace('USDT','');
    const fr = s.funding_rate!=null?(s.funding_rate*100).toFixed(4)+'%':'—';
    return `<div class="pcard${ai?' aipick':''}" data-sym="${s.symbol}">
      <div class="pcard-top">
        <span class="psym">${base}<span style="color:var(--tx3);font-weight:400">/USDT</span></span>
        <span class="pchg ${cc}">${fPct(chg)}</span>
      </div>
      <div class="pprc">${fP(s.price)}</div>
      <div class="pmeta">
        <span>Vol ${fVol(s.turnover_24h)}</span>
        <span>OI ${fVol(s.open_interest)}</span>
        <span>FR ${fr}</span>
        <span>H ${fP(s.high_24h)} · L ${fP(s.low_24h)}</span>
      </div>
      <div class="pspark" data-spark="${s.symbol}" id="sp-${s.symbol}"></div>
    </div>`;
  }).join('');

  grid.querySelectorAll('.pcard').forEach(c => c.addEventListener('click', () => {
    S.sym = c.dataset.sym; goPage('chart');
  }));

  // Lazy sparklines via IntersectionObserver
  if (S.sparkObs) S.sparkObs.disconnect();
  S.sparkObs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const sym = e.target.dataset.spark;
      if (sym && !S.sparkLoaded.has(sym)) { S.sparkLoaded.add(sym); drawSpark(sym); }
      S.sparkObs.unobserve(e.target);
    });
  }, { root: grid, rootMargin: '60px' });
  grid.querySelectorAll('[data-spark]').forEach(el => S.sparkObs.observe(el));
}

async function drawSpark(sym) {
  const d = await get(`/api/klines/${sym}?tf=1h&bars=48`);
  const el = $(`sp-${sym}`);
  if (!el || !d?.candles?.length) return;
  const cls = d.candles.map(c => c.close);
  const w = el.clientWidth || 200, h = 32;
  const mn = Math.min(...cls), mx = Math.max(...cls), rng = mx-mn||1;
  const step = w/(cls.length-1);
  const pts = cls.map((c,i) => `${i*step},${h-((c-mn)/rng)*(h-4)-2}`).join(' ');
  const up = cls[cls.length-1] >= cls[0];
  const col = up ? '#1db88c' : '#e85555';
  el.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.5" opacity=".85"/></svg>`;
}

// Market controls
$('msearch')?.addEventListener('input', renderMarket);
qsa('.tab[data-mf]').forEach(b => b.addEventListener('click', () => {
  qsa('.tab[data-mf]').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); renderMarket();
}));
$('mrefresh')?.addEventListener('click', async () => {
  S.universe = []; S.sparkLoaded.clear(); renderSkeletons();
  await fetchUniverse(); renderMarket();
});

// ── Chart ────────────────────────────────────────────────────────────────────
async function showChart() {
  await nf(); await nf(); // wait for page to be visible before measuring
  buildChartSidebar();
  initChart();
  await loadChartData();
  loadFlags();
}

function initChart() {
  const box = $('chartbox');
  if (!box || S.chart) {
    // Already exists — just resize in case layout changed
    if (S.chart) S.chart.applyOptions({ width: box.clientWidth, height: box.clientHeight });
    return;
  }
  S.chart = LightweightCharts.createChart(box, {
    width: box.clientWidth, height: box.clientHeight,
    layout: { background:{type:'solid',color:'#111c28'}, textColor:'#6b84a0', fontFamily:"'Inter',sans-serif", fontSize:12 },
    grid: { vertLines:{color:'#1d2e42'}, horzLines:{color:'#1d2e42'} },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine:{color:'#3f9fe8',width:1,style:2,labelBackgroundColor:'#3f9fe8'},
      horzLine:{color:'#3f9fe8',width:1,style:2,labelBackgroundColor:'#3f9fe8'},
    },
    rightPriceScale: { borderColor:'#1d2e42', scaleMargins:{top:0.05,bottom:0.22} },
    timeScale: { borderColor:'#1d2e42', timeVisible:true, secondsVisible:false },
  });
  S.cSeries = S.chart.addCandlestickSeries({
    upColor:'#1db88c', downColor:'#e85555',
    borderUpColor:'#1db88c', borderDownColor:'#e85555',
    wickUpColor:'#1db88c', wickDownColor:'#e85555',
  });
  S.vSeries = S.chart.addHistogramSeries({ priceFormat:{type:'volume'}, priceScaleId:'vol' });
  S.chart.priceScale('vol').applyOptions({ scaleMargins:{top:0.82,bottom:0} });
  S.cRO = new ResizeObserver(() => {
    if (S.chart && box.clientWidth) S.chart.applyOptions({ width:box.clientWidth, height:box.clientHeight });
  });
  S.cRO.observe(box);
}

async function loadChartData() {
  const sym = S.sym, tf = S.tf;
  // Update header
  $('csym').textContent = sym;
  $('cprc').textContent = '…';
  $('cchg').textContent = '';
  $('cai').textContent = S.aiPicks.has(sym) ? '★ AI PICK' : '';
  qsa('.tfb').forEach(b => b.classList.toggle('active', b.dataset.tf===tf));
  syncChartSidebarActive(sym);

  const d = await get(`/api/klines/${sym}?tf=${tf}&bars=300`);
  if (!d?.candles?.length) { console.warn('No candles for', sym, tf); return; }

  S.cSeries.setData(d.candles.map(c => ({ time:c.time, open:c.open, high:c.high, low:c.low, close:c.close })));
  S.vSeries.setData(d.candles.map(c => ({
    time:c.time, value:c.volume,
    color: c.close>=c.open ? 'rgba(29,184,140,.3)' : 'rgba(232,85,85,.3)',
  })));
  S.chart.timeScale().fitContent();

  const last = d.candles[d.candles.length-1];
  $('cprc').textContent = fP(last.close);
  const uni = S.universe.find(u => u.symbol===sym);
  if (uni?.change_24h != null) {
    const el = $('cchg');
    el.textContent = fPct(uni.change_24h);
    el.className = 'cchg ' + (uni.change_24h>=0?'up':'dn');
  }
}

function buildChartSidebar() {
  if (!S.universe.length) return; // will rebuild when data arrives
  renderChartSidebar('');
}

function renderChartSidebar(q) {
  const box = $('cpairs');
  if (!box) return;
  let list = S.universe;
  if (q) list = list.filter(s => s.symbol.toLowerCase().includes(q));
  box.innerHTML = list.map(s => {
    const ai = S.aiPicks.has(s.symbol);
    const act = s.symbol === S.sym;
    const chg = s.change_24h, cc = chg!=null?(chg>=0?'up':'dn'):'';
    return `<div class="cpair${act?' active':''}${ai&&!act?' aipick':''}" data-sym="${s.symbol}">
      <span class="cp-sym">${s.symbol.replace('USDT','')}</span>
      <span class="cp-chg ${cc}">${fPct(chg)}</span>
    </div>`;
  }).join('');
  box.querySelectorAll('.cpair').forEach(el => el.addEventListener('click', () => {
    S.sym = el.dataset.sym;
    loadChartData();
    loadFlags();
  }));
}

function syncChartSidebarActive(sym) {
  $('cpairs')?.querySelectorAll('.cpair').forEach(el => {
    const isAct = el.dataset.sym === sym;
    el.classList.toggle('active', isAct);
    if (isAct) el.classList.remove('aipick');
    else if (S.aiPicks.has(el.dataset.sym)) el.classList.add('aipick');
  });
}

$('csearch')?.addEventListener('input', e => renderChartSidebar(e.target.value.toLowerCase()));
qsa('.tfb').forEach(b => b.addEventListener('click', () => {
  S.tf = b.dataset.tf;
  loadChartData();
}));

// ── Flags ────────────────────────────────────────────────────────────────────
const FLAG_COLOR = { green:'flag-green', red:'flag-red', gold:'flag-gold', orange:'flag-orange', blue:'flag-blue', purple:'flag-purple' };

async function loadFlags() {
  const sym = S.sym;
  $('flags-sym').textContent = sym;
  $('flags-grid').innerHTML = '<div class="loading-sm">Computing flags…</div>';
  $('ind-row').innerHTML = '';

  const d = await get(`/api/flags/${sym}`);
  if (!d || d.error) {
    $('flags-grid').innerHTML = `<div class="loading-sm">${d?.error || 'Failed'}</div>`;
    return;
  }

  // Render flag pills
  $('flags-grid').innerHTML = (d.flags || []).map(f => {
    const cls = FLAG_COLOR[f.color] || 'flag-blue';
    const rate = f.fire_rate > 0 ? ` · ${f.fire_rate}%` : '';
    return `<div class="flag-pill ${cls}${f.active?' on':''}" title="${f.key}${rate}">${f.label}${f.active?'':rate}</div>`;
  }).join('') || '<div class="loading-sm">No flags available</div>';

  // Render indicator chips
  const IND_NAMES = {
    rsi_14:'RSI', adx_14:'ADX', macd_hist:'MACD', supertrend_dir:'ST',
    bb_width_20:'BBW', ema_21:'EMA21', ema_50:'EMA50', atr_14:'ATR',
  };
  $('ind-row').innerHTML = Object.entries(d.indicators || {}).map(([k,v]) =>
    `<div class="ind-chip">${IND_NAMES[k]||k} <b>${k==='supertrend_dir'?(v>0?'▲BULL':'▼BEAR'):v}</b></div>`
  ).join('');
}

// ── AI Calls ─────────────────────────────────────────────────────────────────
async function loadCalls() {
  const body = $('ctbody');
  body.innerHTML = '<tr><td colspan="8"><div class="loading"><div class="spinner"></div></div></td></tr>';
  let url = `/api/ai/calls?page=${S.callsPage}&per_page=25`;
  if (S.callsFilter) url += `&call_type=${S.callsFilter}`;
  const d = await get(url);
  if (!d?.calls?.length) {
    body.innerHTML = '<tr><td colspan="8"><div class="empty">No AI calls recorded yet.<br>Run the trading engine to generate logs.</div></td></tr>';
    renderPager(0,1); return;
  }
  body.innerHTML = d.calls.map(c => {
    const tc = (c.call_type||'').includes('social')?'social':(c.call_type||'').includes('review')?'review':'analysis';
    const tok = c.tokens_in!=null?`${c.tokens_in}→${c.tokens_out||'?'}`:'—';
    return `<tr data-id="${c.call_id}">
      <td style="white-space:nowrap;color:var(--tx2)">${fT(c.ts_utc)}</td>
      <td><span class="cbadge ${tc}">${esc(c.call_type)}</span></td>
      <td style="color:var(--tx2);font-size:.75rem">${esc((c.model||'').split('/').pop())}</td>
      <td style="font-weight:600">${esc(c.symbol||'—')}</td>
      <td style="font-family:var(--mono);font-size:.74rem">${tok}</td>
      <td style="font-family:var(--mono);color:var(--gd)">${c.cost_usd!=null?'$'+c.cost_usd.toFixed(5):'—'}</td>
      <td style="font-family:var(--mono);font-size:.74rem">${c.latency_ms!=null?c.latency_ms+'ms':'—'}</td>
      <td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--tx3);font-size:.74rem">${esc(c.response_preview)}</td>
    </tr>`;
  }).join('');
  body.querySelectorAll('tr[data-id]').forEach(tr => tr.addEventListener('click', () => openModal(tr.dataset.id)));
  renderPager(d.total, d.page);
}

function renderPager(total, cur) {
  const el = $('pager');
  const pages = Math.ceil(total/25)||1;
  el.innerHTML = `<button id="pp" ${cur<=1?'disabled':''}>← Prev</button><span>Page ${cur}/${pages} · ${total} calls</span><button id="pn" ${cur>=pages?'disabled':''}>Next →</button>`;
  $('pp')?.addEventListener('click', ()=>{ S.callsPage--; loadCalls(); });
  $('pn')?.addEventListener('click', ()=>{ S.callsPage++; loadCalls(); });
}

qsa('.tab[data-cf]').forEach(b => b.addEventListener('click', () => {
  qsa('.tab[data-cf]').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); S.callsFilter = b.dataset.cf; S.callsPage = 1; loadCalls();
}));

// ── Modal ────────────────────────────────────────────────────────────────────
async function openModal(id) {
  $('mbg').classList.add('open');
  $('mbody').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  $('mtags').innerHTML = '';
  const d = await get(`/api/ai/calls/${id}`);
  if (!d || d.error) { $('mbody').innerHTML = '<div class="empty">Call not found</div>'; return; }
  const resp = d.response||{}, usage = resp.usage||{};
  const tc = (d.call_type||'').includes('social')?'social':(d.call_type||'').includes('review')?'review':'analysis';
  $('mtags').innerHTML = `
    <span class="cbadge ${tc}">${esc(d.call_type)}</span>
    <span style="font-weight:600">${esc(d.symbol||'—')}</span>
    <span style="color:var(--tx2);font-size:.78rem">${esc((d.model||'').split('/').pop())}</span>
    <span style="font-family:var(--mono);font-size:.76rem;color:var(--gd)">${d.cost_usd!=null?'$'+d.cost_usd.toFixed(5):'—'}</span>
    <span style="font-family:var(--mono);font-size:.76rem;color:var(--tx2)">${usage.prompt_tokens||'?'}→${usage.completion_tokens||'?'} tok · ${resp.latency_ms||'?'}ms</span>`;
  let html = '<p class="sec-title">Messages</p>';
  (d.messages||[]).forEach(m => {
    html += `<div class="msg"><div class="msg-role ${m.role}">${m.role}</div><div class="msg-body">${esc(m.content)}</div></div>`;
  });
  if (resp.content) {
    html += '<p class="sec-title" style="margin-top:14px">Response</p>';
    html += `<div class="msg"><div class="msg-role assistant">assistant</div><div class="msg-body">${esc(resp.content)}</div></div>`;
  }
  if (d.error) html += `<div style="margin-top:12px;padding:11px;background:var(--rdb);border-radius:var(--r);color:var(--rd);font-family:var(--mono);font-size:.76rem">${esc(d.error)}</div>`;
  $('mbody').innerHTML = html;
}
$('mx').addEventListener('click', () => $('mbg').classList.remove('open'));
$('mbg').addEventListener('click', e => { if (e.target===e.currentTarget) $('mbg').classList.remove('open'); });

// ── Status ───────────────────────────────────────────────────────────────────
async function loadStatus() {
  $('sgrid').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  const d = await get('/api/status');
  if (!d) { $('sgrid').innerHTML = '<div class="empty">Connection error</div>'; return; }
  const c = d.config||{}, r = d.risk||{};
  const spent = c.today_spent||0, budget = c.daily_budget||1;
  $('sgrid').innerHTML = [
    ['Status','● Online','gr'], ['AI Calls Today',d.ai_calls_today||0,''],
    ['Budget',`$${budget.toFixed(2)}`,''], ['Spent Today',`$${spent.toFixed(4)} (${(spent/budget*100).toFixed(1)}%)`,'gd'],
    ['Paper Equity',`$${(r.equity_usd||0).toLocaleString()}`,'gr'], ['Risk/Trade',`${r.risk_per_trade_pct}%`,''],
    ['Max Positions',r.max_positions,''], ['SL ATR Mult',`${r.atr_mult_sl}×`,''],
    ['Universe Size',c.universe_size||'?',''], ['AI Mode',c.ai_dry_run?'⚠ DRY RUN':'● LIVE',c.ai_dry_run?'gd':'gr'],
    ['Social Model',c.model_social||'?','sm'], ['Analysis Model',c.model_deep||'?','sm'], ['Review Model',c.model_review||'?','sm'],
  ].map(([l,v,cls]) => `<div class="scard"><div class="scard-lbl">${l}</div><div class="scard-val ${cls}">${v}</div></div>`).join('');
}

// ── Status bar ───────────────────────────────────────────────────────────────
async function refreshBar() {
  const d = await get('/api/status');
  const dot = $('sdot'), meta = $('smeta');
  if (d) {
    dot.style.cssText = 'background:#1db88c;box-shadow:0 0 6px #1db88c';
    meta.textContent = `${d.ai_calls_today||0} calls · $${(d.config?.today_spent||0).toFixed(4)}`;
  } else {
    dot.style.cssText = 'background:#e85555;box-shadow:0 0 6px #e85555';
    meta.textContent = 'Offline';
  }
}

// Clock
function tick() {
  const el = $('tbar-time');
  if (el) el.textContent = new Date().toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})+' UTC';
}
setInterval(tick, 1000); tick();

// ── Keyboard shortcuts ───────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key==='Escape') $('mbg').classList.remove('open');
  if (!e.ctrlKey && !e.metaKey && !e.altKey) {
    if (e.key==='1') goPage('market');
    if (e.key==='2') goPage('chart');
    if (e.key==='3') goPage('calls');
    if (e.key==='4') goPage('status');
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────
(async function init() {
  await refreshBar();
  setInterval(refreshBar, 30000);
  // Pre-load universe then show market page
  await fetchUniverse();
  goPage('market');
  if (!S.universe.length) {
    setTimeout(async () => {
      await fetchUniverse();
      if (S.page === 'market') renderMarket();
    }, 3000);
  }
})();

// Expose for app2.js
window.S = S;
window.get = get;
window.goPage = goPage;
window.loadChartData = loadChartData;
