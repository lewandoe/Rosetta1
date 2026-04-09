
const invoke = window.__TAURI__.core.invoke;
const listen = window.__TAURI__.event.listen;
const getCurrentWindow = () => window.__TAURI__.window.getCurrentWindow();

/**
 * app.js — Rosetta1 frontend
 * ES module; import map in index.html resolves the bare specifiers below
 * to the esm.sh CDN builds, which call window.__TAURI_INTERNALS__ injected by Tauri v2.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────────────────────
const S = {
  running:      false,
  mode:         'paper',
  symbols:      ['SPY','QQQ','TSLA','NVDA','AAPL','MSFT','GOOGL','AMZN','META','AMD'],
  logLevel:     'INFO',
  errorCount:   0,
  dayTrades:    0,
  maxDailyLoss: 200,
  perfRange:    'today',
  unlisten:     null,
  eqChart:      null,
  sigChart:     null,
  posTimer:     null,
  dashTimer:    null,
  tradeTimer:   null,
  logs:         [],
  trades:       [],
  defaults: {
    max_daily_loss: 200, max_open_positions: 3,
    max_capital_per_trade_pct: 0.10, max_stop_distance_pct: 0.02,
    min_confidence_score: 70, eod_liquidation_hour: 15, eod_liquidation_minute: 45,
    log_level: 'INFO',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
const $  = (s, ctx = document) => ctx.querySelector(s);
const $$ = (s, ctx = document) => [...ctx.querySelectorAll(s)];

const fmt$ = (n, dec = 2) => {
  if (n == null || isNaN(+n)) return '$0.00';
  const v = Math.abs(+n).toFixed(dec);
  return (+n >= 0 ? '+' : '-') + '$' + v;
};
const fmtPct = n => n == null ? '0.0%' : ((+n) * 100).toFixed(1) + '%';
const fmtHold = s => {
  if (s == null) return '—';
  s = Math.round(+s);
  if (s < 60)   return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
};
const elapsed = iso => {
  if (!iso) return '—';
  const t = new Date(iso.includes('Z') ? iso : iso + 'Z');
  return fmtHold(Math.floor((Date.now() - t) / 1000));
};
const todayISO = () => new Date().toISOString().slice(0, 10);
const ts       = () => new Date().toLocaleTimeString('en-US', { hour12: false });
const esc      = s  => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const pnlCls   = v  => v == null ? '' : +v >= 0 ? 'green' : 'red';

const badgeHtml = r => {
  if (!r) return `<span class="badge bed">—</span>`;
  switch (r.toUpperCase()) {
    case 'TP_HIT': return `<span class="badge btp">TP_HIT</span>`;
    case 'SL_HIT': return `<span class="badge bsl">SL_HIT</span>`;
    case 'MANUAL': return `<span class="badge bma">MANUAL</span>`;
    case 'EOD':    return `<span class="badge bed">EOD</span>`;
    default:       return `<span class="badge bed">${esc(r)}</span>`;
  }
};

const isMarketOpen = () => {
  const et  = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const m   = et.getHours() * 60 + et.getMinutes();
  const day = et.getDay();
  return day > 0 && day < 6 && m >= 570 && m < 960; // 9:30–16:00 ET
};

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────
function showSec(name) {
  $$('.nb[data-sec]').forEach(b => b.classList.toggle('active', b.dataset.sec === name));
  $$('.sec').forEach(s => s.classList.toggle('active', s.id === `sec-${name}`));
  if (name === 'dashboard')   onDashboardVisible();
  if (name === 'tradelog')    refreshTrades();
  if (name === 'performance') loadPerformance();
  if (name === 'settings')    loadSettings();
  if (name === 'signals')     initSigGrid();
}

// ─────────────────────────────────────────────────────────────────────────────
// Titlebar
// ─────────────────────────────────────────────────────────────────────────────
function initTitlebar() {
  const win = getCurrentWindow();
  $('#wc-min').addEventListener('click',  ()  => win.minimize());
  $('#wc-max').addEventListener('click',  async () => (await win.isMaximized()) ? win.unmaximize() : win.maximize());
  $('#wc-close').addEventListener('click', () => win.close());
}

function updateMarket() {
  const el = $('#tb-market');
  if (isMarketOpen()) { el.textContent = '● MARKET OPEN'; el.className = 'mkt-status open'; }
  else                { el.textContent = '● MARKET CLOSED'; el.className = 'mkt-status'; }
}

function setModeBadge(mode) {
  const el = $('#tb-mode');
  el.textContent = mode.toUpperCase();
  el.className   = `mode-badge ${mode}`;
}

function addError() {
  S.errorCount++;
  const el = $('#tb-errors');
  el.textContent = `${S.errorCount} ERR`;
  el.classList.remove('hidden');
}

function setHaltBadge(show) { $('#tb-halt').classList.toggle('hidden', !show); }

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────
async function runChecks() {
  let r;
  try { r = await invoke('check_python'); }
  catch (e) { appendLog(`check_python error: ${e}`, 'error'); return false; }

  setChk('chk-venv', r.venv_found,
    'Run: <code>python3 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt</code>');
  setChk('chk-env', r.env_found,
    'Run: <code>cp .env.example .env</code> then fill in your credentials.');
  setChk('chk-db', r.db_found,
    'Run: <code>mkdir -p /Users/eric/Rosetta1/db</code>');

  const ok = r.venv_found && r.env_found && r.db_found;
  syncStartBtn(ok);
  return ok;
}

function setChk(id, ok, fixHtml) {
  const row  = $(`#${id}`);
  const icon = $('.chk-icon', row);
  const fix  = $('.chk-fix', row);
  icon.textContent = ok ? '✓' : '✗';
  icon.className   = `chk-icon ${ok ? 'ok' : 'fail'}`;
  ok ? fix.classList.add('hidden') : (fix.innerHTML = fixHtml, fix.classList.remove('hidden'));
}

function syncStartBtn(checksOk) {
  const btn = $('#btn-start');
  const err = $('#start-err');
  const noSym = S.symbols.length === 0;
  if (!checksOk) {
    btn.disabled = true; err.textContent = 'Fix the issues above before starting.'; err.classList.remove('hidden');
  } else if (noSym) {
    btn.disabled = true; err.textContent = 'Select at least one symbol.'; err.classList.remove('hidden');
  } else {
    btn.disabled = false; err.classList.add('hidden');
  }
}

function initSetup() {
  // Mode
  $$('.mode-btn').forEach(b => b.addEventListener('click', () => {
    S.mode = b.dataset.mode;
    $$('.mode-btn').forEach(x => x.classList.toggle('active', x.dataset.mode === S.mode));
    $('#live-warn').classList.toggle('hidden', S.mode !== 'live');
    setModeBadge(S.mode);
  }));

  // Symbols
  $$('.pill').forEach(p => p.addEventListener('click', () => {
    const sym = p.dataset.sym;
    p.classList.toggle('active');
    S.symbols = p.classList.contains('active')
      ? [...S.symbols, sym]
      : S.symbols.filter(s => s !== sym);
    syncStartBtn(true);
  }));
  $('#btn-selall').addEventListener('click', () => {
    S.symbols = [];
    $$('.pill').forEach(p => { p.classList.add('active'); S.symbols.push(p.dataset.sym); });
    syncStartBtn(true);
  });
  $('#btn-clrall').addEventListener('click', () => {
    S.symbols = [];
    $$('.pill').forEach(p => p.classList.remove('active'));
    syncStartBtn(true);
  });

  // Log level
  $('#log-lvl').addEventListener('change', e => { S.logLevel = e.target.value; });

  // Recheck
  $('#btn-recheck').addEventListener('click', runChecks);

  // Start
  $('#btn-start').addEventListener('click', startSession);
}

async function startSession() {
  const btn = $('#btn-start');
  btn.disabled = true;
  $('#start-lbl').classList.add('hidden');
  $('#start-spin').classList.remove('hidden');

  try {
    await invoke('start_bot', { mode: S.mode, symbols: S.symbols, logLevel: S.logLevel });
    S.running = true; S.errorCount = 0; S.dayTrades = 0;
    onBotStarted();
    showSec('dashboard');
  } catch (e) {
    $('#start-err').textContent = `Failed to start: ${e}`;
    $('#start-err').classList.remove('hidden');
    btn.disabled = false;
  } finally {
    $('#start-lbl').classList.remove('hidden');
    $('#start-spin').classList.add('hidden');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bot lifecycle
// ─────────────────────────────────────────────────────────────────────────────
function onBotStarted() {
  setModeBadge(S.mode);
  $('#nb-stop').classList.remove('hidden');
  startPolling();
}

function onBotStopped() {
  S.running = false;
  $('#nb-stop').classList.add('hidden');
  stopPolling();
}

async function stopBot() {
  if (!S.running) return;
  try { await invoke('stop_bot'); onBotStopped(); }
  catch (e) { appendLog(`stop_bot error: ${e}`, 'error'); }
}

function startPolling() {
  S.posTimer   = setInterval(refreshPositions, 2000);
  S.dashTimer  = setInterval(refreshDashboard, 5000);
  S.tradeTimer = setInterval(refreshTrades,    5000);
  refreshPositions(); refreshDashboard(); refreshTrades();
}
function stopPolling() {
  clearInterval(S.posTimer);
  clearInterval(S.dashTimer);
  clearInterval(S.tradeTimer);
}
function onDashboardVisible() {
  if (S.running) { refreshPositions(); refreshDashboard(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard
// ─────────────────────────────────────────────────────────────────────────────
async function refreshDashboard() {
  let sum;
  try { sum = await invoke('get_account_summary'); } catch { return; }

  const pnlEl = $('#m-pnl');
  pnlEl.textContent = fmt$(sum.today_pnl);
  pnlEl.className   = `mc-val mono ${pnlCls(sum.today_pnl)}`;

  $('#m-wr').textContent  = fmtPct(sum.win_rate);
  $('#m-pos').textContent = `${sum.open_positions} / ${S.defaults.max_open_positions}`;

  const loss   = Math.max(0, -(sum.today_pnl));
  const remain = Math.max(0, S.maxDailyLoss - loss);
  const pct    = Math.min(100, (loss / S.maxDailyLoss) * 100);

  $('#m-dlr').textContent  = `$${remain.toFixed(2)}`;
  $('#loss-used').textContent = `$${loss.toFixed(2)} used`;
  $('#loss-rem').textContent  = `$${remain.toFixed(2)} remaining`;

  const bar = $('#loss-bar');
  bar.style.width = pct + '%';
  bar.className   = 'prog-fill' + (pct >= 80 ? ' danger' : pct >= 50 ? ' warn' : '');

  updatePDT(S.dayTrades);
}

function updatePDT(n) {
  $('#pdt-lbl').textContent = `${n} of 3 day trades used (rolling 5 days)`;
  for (let i = 0; i < 3; i++) {
    const d = $(`#dot-${i}`);
    d.className = 'dot' + (i < n ? (n >= 3 ? ' maxed' : ' used') : '');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Open Positions
// ─────────────────────────────────────────────────────────────────────────────
async function refreshPositions() {
  let pos;
  try { pos = await invoke('get_open_positions'); } catch { return; }

  const body = $('#pos-body');
  if (!pos || pos.length === 0) {
    body.innerHTML = '<tr class="erow"><td colspan="8">No open positions — scanning for signals</td></tr>';
    return;
  }
  body.innerHTML = pos.map(p => `
    <tr>
      <td class="mono">${p.symbol}</td>
      <td><span class="sig-dir long" style="margin:0">LONG</span></td>
      <td class="mono">${p.signal_type || '—'}</td>
      <td class="mono">$${p.entry_price != null ? (+p.entry_price).toFixed(2) : '—'}</td>
      <td class="mono muted">—</td>
      <td class="mono muted">—</td>
      <td class="mono muted" data-entry="${p.entry_time || ''}">${elapsed(p.entry_time)}</td>
      <td><button class="btn-dng" onclick="closePos('${p.symbol}')">Close</button></td>
    </tr>`).join('');
}

// Make closePos available globally (called from inline onclick)
window.closePos = async function(sym) {
  try {
    await invoke('stop_position', { symbol: sym });
    appendLog(`Close request queued for ${sym}`, 'warn');
  } catch (e) { appendLog(`stop_position error: ${e}`, 'error'); }
};

// Live hold-time ticks without re-fetching
setInterval(() => {
  $$('#pos-body [data-entry]').forEach(el => { el.textContent = elapsed(el.dataset.entry); });
}, 1000);

// ─────────────────────────────────────────────────────────────────────────────
// Signal Scanner
// ─────────────────────────────────────────────────────────────────────────────
const ALL_SYMS = ['SPY','QQQ','TSLA','NVDA','AAPL','MSFT','GOOGL','AMZN','META','AMD'];

function initSigGrid() {
  const grid = $('#sig-grid');
  grid.innerHTML = ALL_SYMS.map(sym => {
    const active = S.symbols.includes(sym) ? '' : ' inactive';
    return `<div class="sig-card${active}" id="sc-${sym}">
      <div class="sig-sym">${sym}</div>
      <div class="sig-type" id="st-${sym}">scanning…</div>
      <div class="sig-dir none" id="sd-${sym}">—</div>
      <div class="sig-bar"><div class="sig-fill" id="sf-${sym}" style="width:0%"></div></div>
      <div class="sig-num"  id="sn-${sym}">0</div>
    </div>`;
  }).join('');
}

function updateSigCard(sym, type, dir, conf) {
  const card = $(`#sc-${sym}`); if (!card) return;
  $(`#st-${sym}`).textContent = type || 'scanning…';
  const dirEl = $(`#sd-${sym}`);
  const d     = (dir || '').toUpperCase();
  dirEl.textContent = d || '—';
  dirEl.className   = 'sig-dir' + (d==='LONG' ? ' long' : d==='SHORT' ? ' short' : ' none');
  $(`#sf-${sym}`).style.width = (conf || 0) + '%';
  $(`#sn-${sym}`).textContent = conf || 0;
  card.classList.add('firing');
  setTimeout(() => card.classList.remove('firing'), 800);
}

// ─────────────────────────────────────────────────────────────────────────────
// Trade Log
// ─────────────────────────────────────────────────────────────────────────────
async function refreshTrades() {
  let trades;
  try { trades = await invoke('get_trades', { date: todayISO() }); } catch { return; }
  S.trades = trades || [];
  renderTrades(S.trades);
}

function renderTrades(trades) {
  const body = $('#trade-body');
  const foot = $('#trade-foot');
  if (!trades || trades.length === 0) {
    body.innerHTML = '<tr class="erow"><td colspan="9">No trades recorded today</td></tr>';
    foot.classList.add('hidden'); return;
  }
  body.innerHTML = trades.map(t => {
    const time = t.entry_time ? t.entry_time.slice(11, 19) : '—';
    const pnl  = t.gross_pnl != null ? +t.gross_pnl : null;
    return `<tr>
      <td class="mono muted">${time}</td>
      <td class="mono">${t.symbol}</td>
      <td class="mono">${t.signal_type || '—'}</td>
      <td class="mono">$${t.entry_price != null ? (+t.entry_price).toFixed(2) : '—'}</td>
      <td class="mono">$${t.exit_price  != null ? (+t.exit_price).toFixed(2)  : '—'}</td>
      <td class="mono">${t.shares != null ? t.shares : '—'}</td>
      <td class="mono ${pnlCls(pnl)}">${pnl != null ? fmt$(pnl) : '—'}</td>
      <td class="mono muted">${fmtHold(t.hold_duration_seconds)}</td>
      <td>${badgeHtml(t.exit_reason)}</td>
    </tr>`;
  }).join('');

  const totPnl = trades.reduce((a, t) => a + (t.gross_pnl || 0), 0);
  const wins   = trades.filter(t => (t.gross_pnl || 0) > 0).length;
  const totEl  = $('#tot-pnl');
  totEl.textContent = fmt$(totPnl);
  totEl.className   = `mono ${pnlCls(totPnl)}`;
  $('#tot-wl').textContent = `${wins}W / ${trades.length - wins}L`;
  foot.classList.remove('hidden');
}

function exportCSV() {
  if (!S.trades.length) return;
  const hdr  = ['Time','Symbol','Signal','Entry','Exit','Shares','PnL','Hold(s)','Reason'];
  const rows = S.trades.map(t => [
    t.entry_time||'', t.symbol, t.signal_type||'',
    t.entry_price!=null?(+t.entry_price).toFixed(2):'',
    t.exit_price !=null?(+t.exit_price).toFixed(2) :'',
    t.shares!=null?t.shares:'',
    t.gross_pnl !=null?(+t.gross_pnl).toFixed(2)  :'',
    t.hold_duration_seconds!=null?t.hold_duration_seconds:'',
    t.exit_reason||'',
  ].map(v=>`"${String(v).replace(/"/g,'""')}"`).join(','));

  const url = URL.createObjectURL(new Blob([[hdr.join(','),...rows].join('\n')],{type:'text/csv'}));
  Object.assign(document.createElement('a'), { href: url, download: `rosetta1_${todayISO()}.csv` }).click();
  URL.revokeObjectURL(url);
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance
// ─────────────────────────────────────────────────────────────────────────────
async function loadPerformance() {
  let d;
  try { d = await invoke('get_performance', { range: S.perfRange }); }
  catch { return; }

  const pEl = $('#p-pnl');
  pEl.textContent = fmt$(d.total_pnl);
  pEl.className   = `mc-val mono ${pnlCls(d.total_pnl)}`;
  $('#p-sharpe').textContent = (+d.sharpe_ratio || 0).toFixed(2);
  $('#p-wr').textContent     = fmtPct(d.win_rate);
  $('#p-dd').textContent     = `$${(+d.max_drawdown || 0).toFixed(2)}`;

  buildEquityChart(d.equity_curve || []);
  buildSigChart(d.by_signal || []);
}

function buildEquityChart(curve) {
  const ctx  = $('#eq-chart').getContext('2d');
  const vals = curve.map(p => p.v);
  const last = vals[vals.length - 1] ?? 0;
  const col  = last >= 0 ? '#00ff88' : '#ff4444';
  if (S.eqChart) S.eqChart.destroy();
  S.eqChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: vals.map((_, i) => i + 1),
      datasets: [{ data: vals, borderColor: col, backgroundColor: col + '18',
                   borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.3 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid:  { color:'rgba(255,255,255,.05)' },
             ticks: { color:'#888', font:{family:'JetBrains Mono',size:10}, callback:v=>`$${v.toFixed(0)}` },
             border:{ color:'rgba(255,255,255,.08)' } },
      },
    },
  });
}

function buildSigChart(bySig) {
  const ctx   = $('#sig-chart').getContext('2d');
  const SIGS  = ['EMA_CROSS','VWAP_REV','MOMENTUM','ORB','RSI'];
  const lkup  = Object.fromEntries(bySig.map(r => [r.signal_type, r]));
  const wins  = SIGS.map(s => lkup[s]?.wins   ?? 0);
  const loses = SIGS.map(s => lkup[s]?.losses  ?? 0);
  if (S.sigChart) S.sigChart.destroy();
  S.sigChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: SIGS,
      datasets: [
        { label:'Wins',   data:wins,  backgroundColor:'#00ff8844', borderColor:'#00ff88', borderWidth:1 },
        { label:'Losses', data:loses, backgroundColor:'#ff444444', borderColor:'#ff4444', borderWidth:1 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
      plugins: { legend: { display:true, labels:{ color:'#888', font:{family:'JetBrains Mono',size:10} } } },
      scales: {
        x: { grid:{display:false}, ticks:{color:'#888', font:{family:'JetBrains Mono',size:10}}, border:{color:'rgba(255,255,255,.08)'} },
        y: { grid:{color:'rgba(255,255,255,.05)'}, ticks:{color:'#888', font:{family:'JetBrains Mono',size:10}, stepSize:1}, border:{color:'rgba(255,255,255,.08)'} },
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings
// ─────────────────────────────────────────────────────────────────────────────
async function loadSettings() {
  let cfg;
  try { cfg = await invoke('get_settings'); }
  catch { cfg = S.defaults; }

  $('#s-loss').value = cfg.max_daily_loss           || 200;
  $('#s-pos').value  = cfg.max_open_positions        || 3;
  $('#s-cap').value  = ((+cfg.max_capital_per_trade_pct || 0.10) * 100).toFixed(1);
  $('#s-stop').value = ((+cfg.max_stop_distance_pct     || 0.02) * 100).toFixed(1);
  $('#s-conf').value = cfg.min_confidence_score      || 70;
  $('#s-eod').value  = 60 - (+cfg.eod_liquidation_minute || 45);

  S.maxDailyLoss = +cfg.max_daily_loss || 200;
}

async function saveSettings() {
  const payload = {
    max_daily_loss:             $('#s-loss').value,
    max_open_positions:         $('#s-pos').value,
    max_capital_per_trade_pct: (+$('#s-cap').value  / 100).toString(),
    max_stop_distance_pct:     (+$('#s-stop').value / 100).toString(),
    min_confidence_score:       $('#s-conf').value,
    eod_liquidation_minute:     (60 - +$('#s-eod').value).toString(),
  };
  try {
    await invoke('save_settings', { settings: payload });
    S.maxDailyLoss = +payload.max_daily_loss;
    $('#sett-note').classList.remove('hidden');
    appendLog('Settings saved to .env', 'ok');
  } catch (e) { appendLog(`save_settings error: ${e}`, 'error'); }
}

function resetSettings() {
  const d = S.defaults;
  $('#s-loss').value = d.max_daily_loss;
  $('#s-pos').value  = d.max_open_positions;
  $('#s-cap').value  = (d.max_capital_per_trade_pct * 100).toFixed(1);
  $('#s-stop').value = (d.max_stop_distance_pct     * 100).toFixed(1);
  $('#s-conf').value = d.min_confidence_score;
  $('#s-eod').value  = 60 - d.eod_liquidation_minute;
}

// ─────────────────────────────────────────────────────────────────────────────
// System Log
// ─────────────────────────────────────────────────────────────────────────────
function classifyLine(l) {
  const low = l.toLowerCase();
  if (/\b(ok\b|started|seeded|closed|session complete)/.test(low)) return 'ok';
  if (/\b(error|halt|sl_hit)\b/.test(low))                          return 'error';
  if (/\b(warning|warn|rejected|pdt)\b/.test(low))                  return 'warn';
  return 'normal';
}

function appendLog(line, cls = null) {
  cls = cls ?? classifyLine(line);
  S.logs.push({ ts: ts(), line, cls });

  const panel = $('#log-panel');
  const empty = $('.log-empty', panel);
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = `log-line ${cls}`;
  div.innerHTML = `<span class="log-ts">${ts()}</span><span class="log-msg">${esc(line)}</span>`;
  panel.appendChild(div);
  while (panel.children.length > 2000) panel.removeChild(panel.firstChild);
  panel.scrollTop = panel.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bot log parser  (receives "bot_log" events from Rust)
// ─────────────────────────────────────────────────────────────────────────────
function parseLine(line) {
  appendLog(line);

  // "Signal fired: SPY EMA_CROSS"
  const sig = line.match(/Signal fired:\s+(\w+)\s+(\w+)/i);
  if (sig) updateSigCard(sig[1], sig[2], 'LONG', 80);

  // "TRADE CLOSED: SPY +12.50 | 00:03:45 | TP_HIT"
  if (/TRADE CLOSED:/i.test(line) && $('#sec-tradelog').classList.contains('active')) refreshTrades();

  // Error counter
  if (/\bERROR\b/.test(line)) addError();

  // EOD
  if (/EOD \+ all positions closed/i.test(line)) $('#eod-banner').classList.remove('hidden');

  // Session complete
  if (/Session complete/i.test(line)) { showSessionModal(); onBotStopped(); }

  // Trading halted
  if (/trading halted/i.test(line)) setHaltBadge(true);

  // PDT
  if (/day trade/i.test(line)) { S.dayTrades = Math.min(3, S.dayTrades + 1); updatePDT(S.dayTrades); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Session Summary Modal
// ─────────────────────────────────────────────────────────────────────────────
async function showSessionModal() {
  let d;
  try { d = await invoke('get_performance', { range: 'today' }); }
  catch { d = { total_pnl:0, trade_count:0, win_rate:0, sharpe_ratio:0 }; }

  const pnls = S.trades.map(t => t.gross_pnl || 0);
  const best  = pnls.length ? Math.max(...pnls) : 0;
  const worst = pnls.length ? Math.min(...pnls) : 0;

  $('#sm-trades').textContent = d.trade_count || 0;
  $('#sm-wr').textContent     = fmtPct(d.win_rate);
  $('#sm-pnl').textContent    = fmt$(d.total_pnl);
  $('#sm-best').textContent   = fmt$(best);
  $('#sm-worst').textContent  = fmt$(worst);
  $('#sm-sharpe').textContent = (+d.sharpe_ratio || 0).toFixed(2);

  $('#modal-bg').classList.remove('hidden');
}

// ─────────────────────────────────────────────────────────────────────────────
// Emergency halt
// ─────────────────────────────────────────────────────────────────────────────
async function emergencyHalt() {
  if (!confirm('Send emergency halt signal?\n\nThis will signal the Python process to stop trading.')) return;
  try {
    await invoke('emergency_halt');
    setHaltBadge(true);
    appendLog('EMERGENCY HALT signal sent', 'error');
  } catch (e) { appendLog(`emergency_halt error: ${e}`, 'error'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wire all UI events
// ─────────────────────────────────────────────────────────────────────────────
function initEvents() {
  // Sidebar nav
  $$('.nb[data-sec]').forEach(b => b.addEventListener('click', () => showSec(b.dataset.sec)));

  // Bot controls
  $('#nb-stop').addEventListener('click', stopBot);
  $('#nb-halt').addEventListener('click', emergencyHalt);

  // Trade log
  $('#btn-csv').addEventListener('click', exportCSV);

  // Perf range tabs
  $$('.rtab').forEach(t => t.addEventListener('click', () => {
    S.perfRange = t.dataset.range;
    $$('.rtab').forEach(x => x.classList.toggle('active', x.dataset.range === S.perfRange));
    loadPerformance();
  }));

  // Settings
  $('#btn-save-sett').addEventListener('click', saveSettings);
  $('#btn-rst-sett').addEventListener('click', resetSettings);

  // Log
  $('#btn-clrlog').addEventListener('click', () => {
    $('#log-panel').innerHTML = '<div class="log-empty">Log cleared.</div>';
    S.logs = [];
  });
  $('#btn-cpylog').addEventListener('click', () => {
    navigator.clipboard.writeText(S.logs.map(e => `${e.ts}  ${e.line}`).join('\n')).catch(()=>{});
  });

  // Modal
  $('#btn-new-sess').addEventListener('click', () => {
    $('#modal-bg').classList.add('hidden');
    setHaltBadge(false);
    S.errorCount = 0; $('#tb-errors').classList.add('hidden');
    showSec('setup');
  });
  $('#btn-view-perf').addEventListener('click', () => {
    $('#modal-bg').classList.add('hidden');
    showSec('performance');
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────
async function boot() {
  initTitlebar();
  initSetup();
  initEvents();
  initSigGrid();

  // Market clock
  updateMarket();
  setInterval(updateMarket, 60_000);

  // Subscribe to bot log stream
  S.unlisten = await listen('bot_log', ev => parseLine(ev.payload));

  // Run system checks
  await runChecks();

  showSec('setup');
}

boot();
