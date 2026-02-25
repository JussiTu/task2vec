"""
Generates the self-contained interactive UMAP explorer HTML.
Run once after pre-computing .cache/umap_app_data.json.
"""
import json, os, re
from openai import OpenAI


def generate_strategy_narrative(app_data: dict) -> str:
    """Call LLM to write a project-level strategy narrative. Cached in app_data."""
    S = app_data['strategy']

    def _year(s, last=False):
        m = re.findall(r'\d{4}', s or '')
        return (m[-1] if last else m[0]) if m else '?'

    early_year  = _year(S.get('early_period', ''), last=False)
    recent_year = _year(S.get('recent_period', ''), last=True)
    total       = S.get('total_tickets', 0)

    def _top_str(items, n=4):
        out = []
        for t in (items or [])[:n]:
            label = t['label'] if isinstance(t, dict) else str(t)
            pct   = t.get('pct', '') if isinstance(t, dict) else ''
            out.append(f"{label} ({pct}%)" if pct else label)
        return ', '.join(out)

    early_str  = _top_str(S.get('early_top', []))
    recent_str = _top_str(S.get('recent_top', []))

    # Count contributor alignment distribution
    alignments = [
        a.get('umap_alignment', {}).get('umap_cos', None)
        for a in app_data.get('assignees', {}).values()
    ]
    alignments = [x for x in alignments if x is not None]
    n_aligned   = sum(1 for x in alignments if x >= 0.3)
    n_counter   = sum(1 for x in alignments if x <= -0.3)
    n_total_emp = len(alignments)

    prompt = (
        "Write a concise strategic narrative for a software project based on the ticket data below.\n"
        "Requirements:\n"
        "- 6-8 sentences.\n"
        "- Plain English, no jargon about embeddings, UMAP, vectors or cosine.\n"
        "- Describe what the project focused on early, how that changed, and what it focuses on now.\n"
        "- Mention the shift in theme labels by name.\n"
        "- Include one sentence about contributor alignment (how many moved with vs against the project direction).\n"
        "- Do NOT invent facts outside the provided data.\n\n"
        f"Project: Spring (open-source Java framework, Jira tickets)\n"
        f"Period: {early_year} to {recent_year}\n"
        f"Total tickets analysed: {total:,}\n"
        f"Early focus themes: {early_str}\n"
        f"Recent focus themes: {recent_str}\n"
        f"Contributors analysed: {n_total_emp}\n"
        f"Moving with project direction (cos >= 0.3): {n_aligned} contributors\n"
        f"Moving against project direction (cos <= -0.3): {n_counter} contributors\n\n"
        "Narrative:"
    )

    resp = OpenAI().chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.3
    )
    return (resp.choices[0].message.content or '').strip()


with open('.cache/umap_app_data.json', encoding='utf-8') as f:
    app_data = json.load(f)

# Generate strategy narrative (once — cached in JSON)
if not app_data['strategy'].get('narrative'):
    print('Generating strategy narrative via LLM...')
    app_data['strategy']['narrative'] = generate_strategy_narrative(app_data)
    with open('.cache/umap_app_data.json', 'w', encoding='utf-8') as f:
        json.dump(app_data, f, ensure_ascii=False, separators=(',', ':'))
    print('Narrative cached.')

# ── Build strategy panel HTML in Python (avoids all JS string-quoting risks) ──
def _year(s, last=False):
    m = re.findall(r'\d{4}', s or '')
    return (m[-1] if last else m[0]) if m else '?'

def _theme_tags(items):
    out = []
    for t in (items or [])[:4]:
        label = t['label'] if isinstance(t, dict) else str(t)
        out.append(f'<span class="theme-tag">{label}</span>')
    return ''.join(out)

S = app_data['strategy']
_ey = _year(S.get('early_period', ''), last=False)
_ry = _year(S.get('recent_period', ''), last=True)
_total = f"{S.get('total_tickets', 0):,}"
_early_tags  = _theme_tags(S.get('early_top', []))
_recent_tags = _theme_tags(S.get('recent_top', []))

_narrative = S.get('narrative', '')

strategy_html = f"""
<div class="card-id" style="color:#ffd700;margin-bottom:2px">Project Direction</div>
<div class="card-sub">Emergent analysis &middot; {_total} tickets &middot; {_ey}&ndash;{_ry}</div>
<div class="stat-row">
  <div class="stat-box"><div class="val">{_ey}</div><div class="lbl">Earliest</div></div>
  <div class="stat-box"><div class="val">{_ry}</div><div class="lbl">Latest</div></div>
</div>
<div class="sec">Strategic narrative</div>
<div class="narrative">{_narrative}</div>
<div class="sec" style="margin-top:10px">Early focus ({_ey}&ndash;)</div>
<div style="margin-bottom:6px">{_early_tags}</div>
<div class="sec">Recent focus (&ndash;{_ry})</div>
<div style="margin-bottom:16px">{_recent_tags}</div>
<div style="font-size:11px;color:#444;text-align:center;
     border-top:1px solid #1e1e1e;padding-top:14px;line-height:1.8">
  Select an employee above<br>to compare their trajectory<br>with the project direction.
</div>
"""

# Serialize and escape </script> so it can't break the HTML script tag
data_json = json.dumps(app_data, ensure_ascii=False, separators=(',', ':'))
data_json = data_json.replace('</', '<\\/')   # safe JSON escape for forward slashes

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-QTKXE40LPF"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-QTKXE40LPF');
</script>
<title>Spring — Ticket Landscape Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #eee; font-family: 'Segoe UI', sans-serif;
       height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

#header { padding: 10px 18px; background: #161616; border-bottom: 1px solid #2a2a2a;
          display: flex; align-items: center; gap: 18px; flex-wrap: wrap; flex-shrink: 0; }
#header h1 { font-size: 15px; color: #ccc; font-weight: 500; white-space: nowrap; }
#header h1 span { color: #7b68ee; }
select { background: #1e1e1e; color: #eee; border: 1px solid #444; border-radius: 6px;
         padding: 6px 12px; font-size: 13px; cursor: pointer; min-width: 290px; }
select:focus { outline: none; border-color: #7b68ee; }
#stats-bar { font-size: 11px; color: #666; display: flex; gap: 16px; }
#stats-bar b { color: #bbb; }

#main { display: flex; flex: 1; overflow: hidden; min-height: 0; }
#map-col { flex: 1; min-width: 0; display: flex; flex-direction: column; }
#map  { flex: 1; min-height: 0; }
#river-wrap { height: 190px; flex-shrink: 0; border-top: 1px solid #1e1e1e; }
#river { width: 100%; height: 100%; }

#side-col { width: 320px; min-width: 320px; background: #111; border-left: 1px solid #1e1e1e;
            display: flex; flex-direction: column; overflow: hidden; }
#info { flex: 1; padding: 14px 16px; overflow-y: auto; }

#employee-card { display: none; }

.card-id   { font-size: 17px; font-weight: 600; color: #a89cff; margin-bottom: 2px; }
.card-sub  { font-size: 11px; color: #555; margin-bottom: 12px; }
.stat-row  { display: flex; gap: 8px; margin-bottom: 12px; }
.stat-box  { background: #191919; border-radius: 7px; padding: 9px 12px; flex: 1; text-align: center; }
.stat-box .val { font-size: 18px; font-weight: 700; color: #fff; }
.stat-box .lbl { font-size: 9px; color: #555; margin-top: 2px; text-transform: uppercase; letter-spacing: .05em; }
.sec { font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: #444;
       margin: 12px 0 5px; }
.phase-row  { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 4px; }
.phase-chip { border-radius: 4px; padding: 3px 9px; font-size: 10px; font-weight: 600; }
.theme-tag  { display: inline-block; background: #1a1a2e; border: 1px solid #2a2a44;
              border-radius: 4px; padding: 2px 7px; font-size: 10px; color: #99a; margin: 2px 2px 2px 0; }
.align-wrap { background: #1a1a1a; border-radius: 5px; height: 6px; margin: 4px 0 12px; overflow: hidden; }
.align-fill { height: 100%; border-radius: 5px; background: linear-gradient(90deg,#7b68ee,#ffa500); }
.narrative  { font-size: 11px; line-height: 1.75; color: #999; background: #161616;
              border-radius: 7px; padding: 11px 13px; border-left: 3px solid #2a2a2a; }
.arrow-legend { background: #191919; border-radius: 7px; padding: 10px 12px;
                display: flex; flex-direction: column; gap: 10px; margin-bottom: 4px; }
.arrow-row   { display: flex; align-items: flex-start; gap: 10px; }
.arrow-swatch { width: 28px; height: 4px; border-radius: 2px; margin-top: 6px; flex-shrink: 0; }
.arrow-label { font-size: 11px; font-weight: 600; color: #ddd; margin-bottom: 2px; }
.arrow-desc  { font-size: 10px; color: #666; line-height: 1.5; }

/* ── Ticket Advisor ── */
#advisor-wrap { border-top: 1px solid #1e1e1e; padding: 12px 14px; flex-shrink: 0; }
#advisor-toggle { font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
                  color: #555; cursor: pointer; display: flex; align-items: center;
                  justify-content: space-between; user-select: none; }
#advisor-toggle:hover { color: #aaa; }
#advisor-body { display: none; padding-top: 10px; }
#advisor-body.open { display: block; }
textarea#ticket-input { width: 100%; background: #1a1a1a; border: 1px solid #2a2a2a;
                        border-radius: 6px; color: #ddd; font-size: 12px; padding: 8px 10px;
                        resize: vertical; font-family: inherit; min-height: 76px; }
textarea#ticket-input:focus { outline: none; border-color: #7b68ee; }
#analyze-btn { margin-top: 7px; width: 100%; background: #7b68ee; color: #fff;
               border: none; border-radius: 6px; padding: 8px; font-size: 12px;
               font-weight: 600; cursor: pointer; transition: background .2s; }
#analyze-btn:hover:not(:disabled) { background: #6a57dd; }
#analyze-btn:disabled { background: #333; color: #555; cursor: default; }
.adv-advice { font-size: 11px; color: #bbb; line-height: 1.75; background: #161616;
              border-radius: 6px; padding: 10px 12px; border-left: 3px solid #7b68ee;
              white-space: pre-wrap; }
.adv-sim { font-size: 11px; color: #777; margin-bottom: 5px; line-height: 1.5; }
.adv-sim b { color: #aaa; }
.adv-expert-chip { display: inline-block; background: #1a1a2e; border: 1px solid #2a2a44;
                   border-radius: 4px; padding: 2px 7px; font-size: 10px; color: #99a;
                   margin: 2px 2px 0 0; }
</style>
</head>
<body>

<div id="header">
  <h1>Spring — <span>69,156 tickets</span> &nbsp;|&nbsp; Ticket Landscape Explorer</h1>
  <select id="sel" onchange="onSelect(this.value)">
    <option value="">— Select an employee —</option>
  </select>
  <div id="stats-bar" style="display:none">
    Tickets: <b id="sb-t"></b> &nbsp;|&nbsp;
    Active: <b id="sb-y"></b> &nbsp;|&nbsp;
    Alignment: <b id="sb-a"></b>
  </div>
</div>

<div id="main">
  <div id="map-col">
    <div id="map"></div>
    <div id="river-wrap"><div id="river"></div></div>
  </div>
  <div id="side-col">
    <div id="info">
      <!-- Arrow legend: always visible -->
      <div class="sec">Directions on map</div>
      <div class="arrow-legend">
        <div class="arrow-row">
          <div class="arrow-swatch" style="background:#ffd700"></div>
          <div>
            <div class="arrow-label">Project direction</div>
            <div class="arrow-desc" id="c-strat-desc"></div>
          </div>
        </div>
      </div>

      <!-- Ticket Advisor -->
      <div id="advisor-wrap">
        <div id="advisor-toggle" onclick="toggleAdvisor()">
          <span>Analyze a ticket</span>
          <span id="advisor-chevron">&#9658;</span>
        </div>
        <div id="advisor-body">
          <textarea id="ticket-input" placeholder="Paste ticket summary / description here..."></textarea>
          <button id="analyze-btn" onclick="runAnalyze()">Analyze</button>
          <div id="advisor-result"></div>
        </div>
      </div>

      <!-- Strategy panel: shown initially, hidden when employee selected -->
      <div id="placeholder">""" + strategy_html + """</div>
      <div id="employee-card">
        <div class="card-id" id="c-id"></div>
        <div class="card-sub" id="c-sub"></div>
        <div class="stat-row">
          <div class="stat-box"><div class="val" id="c-tc"></div><div class="lbl">Tickets</div></div>
          <div class="stat-box"><div class="val" id="c-yc"></div><div class="lbl">Yrs active</div></div>
          <div class="stat-box"><div class="val" id="c-al"></div><div class="lbl">Alignment</div></div>
        </div>
        <div class="sec">Work story</div>
        <div class="narrative" id="c-narr"></div>
        <div class="sec">Career trajectory</div>
        <div class="narrative" id="c-dir-desc" style="border-left-color:#555;display:none"></div>
        <div class="sec">Alignment with project direction</div>
        <div id="c-align-detail" style="font-size:11px;color:#777;margin-bottom:5px"></div>
        <div class="align-wrap"><div class="align-fill" id="c-abar"></div></div>
        <div class="narrative" id="c-strat-ctx" style="border-left-color:#5a4a00;margin-top:8px;display:none"></div>
        <div class="sec">Career phases</div>
        <div class="phase-row" id="c-phases"></div>
        <div class="sec">Top themes</div>
        <div id="c-themes"></div>
      </div>
    </div>
  </div>
</div>

<script>
const APP = """ + data_json + """;

const PHASE_COLORS = { Early: '#7b68ee', Middle: '#ffa500', Recent: '#ff4444' };
const PHASES = ['Early', 'Middle', 'Recent'];

// ── Populate dropdown ──────────────────────────────────────────────
const selEl = document.getElementById('sel');
APP.assignee_list.forEach(function(row) {
  var aid = row[0], cnt = row[1], sid = row[2];
  var opt = document.createElement('option');
  opt.value = aid;
  opt.textContent = sid + '  (' + cnt.toLocaleString() + ' tickets)';
  selEl.appendChild(opt);
});

// ── Cluster centroids (computed once from sampled bg points) ──────
var clusterCentroids = APP.bg_traces.map(function(t) {
  var sx = 0, sy = 0, n = t.x.length;
  for (var i = 0; i < n; i++) { sx += t.x[i]; sy += t.y[i]; }
  return { x: sx / n, y: sy / n, name: t.name, color: t.color, size: t.size };
});

// ── Cluster label annotations (white text, dark pill background) ───
var clusterAnnotations = clusterCentroids.map(function(c) {
  return {
    x: c.x, y: c.y,
    xref: 'x', yref: 'y',
    text: c.name,
    showarrow: false,
    font: { color: 'white', size: 9 },
    bgcolor: 'rgba(0,0,0,0.6)',
    borderpad: 3,
    opacity: 0.95,
  };
});

// ── Base map layout ────────────────────────────────────────────────
var baseLayout = {
  paper_bgcolor: '#0d0d0d', plot_bgcolor: '#111111',
  margin: { l: 40, r: 10, t: 10, b: 36 },
  showlegend: true,
  legend: {
    bgcolor: 'rgba(13,13,13,0.9)', bordercolor: '#2a2a2a',
    font: { size: 8, color: '#ccc' },
    x: 0.99, y: 0.99, xanchor: 'right', yanchor: 'top',
    tracegroupgap: 2, itemsizing: 'constant',
  },
  xaxis: { showgrid: false, zeroline: false, color: '#333',
           title: { text: 'UMAP-1', font: { color: '#444', size: 10 } } },
  yaxis: { showgrid: false, zeroline: false, color: '#333',
           title: { text: 'UMAP-2', font: { color: '#444', size: 10 } } },
  hovermode: 'closest',
  annotations: clusterAnnotations,
};

var mapCfg = { responsive: true, scrollZoom: true,
               displayModeBar: true, displaylogo: false };

// Invisible centroid markers for hover (show name + count on cursor) ─
function centroidHoverTrace() {
  return {
    type: 'scatter', mode: 'markers',
    x: clusterCentroids.map(function(c) { return c.x; }),
    y: clusterCentroids.map(function(c) { return c.y; }),
    marker: { size: 32, color: 'rgba(0,0,0,0)', opacity: 0.01, line: { width: 0 } },
    customdata: clusterCentroids.map(function(c) { return [c.name, c.size]; }),
    hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[1]:,} tickets<extra></extra>',
    showlegend: false, name: '',
  };
}

// ── Build background traces ────────────────────────────────────────
function bgTraces(opacity) {
  var traces = APP.bg_traces.map(function(t) {
    return {
      type: 'scattergl', mode: 'markers',
      x: t.x, y: t.y, text: t.text,
      marker: { size: 3, color: t.color, opacity: opacity },
      name: t.name + ' (' + t.size.toLocaleString() + ')',
      hovertemplate: '%{text}<extra></extra>',
      legendgroup: 'bg',
    };
  });
  traces.push(centroidHoverTrace());
  return traces;
}

// ── Strategy arrow helpers ─────────────────────────────────────────
var S = APP.strategy;

function strategyArrowAnnotation() {
  return {
    x: S.recent_cent[0], y: S.recent_cent[1],
    ax: S.early_cent[0], ay: S.early_cent[1],
    xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
    showarrow: true, arrowhead: 3, arrowsize: 2.2,
    arrowwidth: 3.5, arrowcolor: '#ffd700',
    opacity: 0.92,
  };
}

function employeeArrowAnnotation(a) {
  var ua = a.umap_alignment;
  if (!ua) return null;
  return {
    x: ua.recent_cent[0], y: ua.recent_cent[1],
    ax: ua.early_cent[0], ay: ua.early_cent[1],
    xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
    showarrow: true, arrowhead: 3, arrowsize: 1.8,
    arrowwidth: 2.5, arrowcolor: 'white',
    opacity: 0.85,
  };
}

// Set static strategy description in side panel
document.getElementById('c-strat-desc').textContent =
  'Where the Spring project moved as a whole — from ' +
  S.early_period.slice(0,4) + ' to ' + S.recent_period.slice(-4) +
  '. Computed from the shift in the centre of all tickets over time.';

// Initial render — full brightness
Plotly.newPlot('map', bgTraces(0.22),
  Object.assign({}, baseLayout, { annotations: clusterAnnotations.concat([strategyArrowAnnotation()]) }),
  mapCfg);

// ── Main select handler ────────────────────────────────────────────
function onSelect(aid) {
  if (!aid) { resetAll(); return; }
  var a = APP.assignees[aid];
  if (!a) return;

  // Stats bar
  var yrs = a.year_range.split('-');
  var yrCount = yrs.length === 2 ? (parseInt(yrs[1]) - parseInt(yrs[0]) + 1) : '?';
  document.getElementById('stats-bar').style.display = '';
  document.getElementById('sb-t').textContent = a.ticket_count.toLocaleString();
  document.getElementById('sb-y').textContent = a.year_range;
  document.getElementById('sb-a').textContent = a.alignment != null ? a.alignment + '%' : '—';

  // Build traces — dim background dots but keep cluster labels faintly visible
  var traces = bgTraces(0.06);

  // Per-phase: dots then hull
  PHASES.forEach(function(ph) {
    var color = PHASE_COLORS[ph];
    var info  = a.phases[ph];
    if (!info) return;

    // Tickets in this phase
    var pts = a.tickets.filter(function(t) { return t.phase === ph; });
    if (!pts.length) return;

    traces.push({
      type: 'scattergl', mode: 'markers',
      x: pts.map(function(t) { return t.x; }),
      y: pts.map(function(t) { return t.y; }),
      text: pts.map(function(t) {
        var cl = APP.cluster_labels[String(t.cluster)];
        var clName = cl ? cl.label : '';
        return '<b>' + t.key + '</b><br>' + t.summary +
               '<br>Year: ' + t.year + ' | ' + ph + '<br>' + clName;
      }),
      marker: { size: 9, color: color, opacity: 0.9,
                line: { width: 0.5, color: 'rgba(255,255,255,0.5)' } },
      name: ph + ' (' + info.years + ')',
      hovertemplate: '%{text}<extra></extra>',
      legendgroup: 'emp',
      legendgrouptitle: { text: ph === 'Early' ? 'Employee ' + a.id : undefined },
    });

  });

  // Phase centroid markers + labels
  PHASES.forEach(function(ph) {
    var info = a.phases[ph];
    if (!info) return;
    traces.push({
      type: 'scatter', mode: 'markers+text',
      x: [info.centroid[0]], y: [info.centroid[1]],
      marker: { size: 20, color: PHASE_COLORS[ph],
                line: { color: 'white', width: 2 } },
      text: [ph + '<br>' + info.years],
      textposition: 'top center',
      textfont: { color: 'white', size: 10 },
      showlegend: false, hoverinfo: 'skip',
    });
  });

  // Arrows: phase transitions + strategy + employee direction
  var annotations = [];
  for (var i = 0; i < PHASES.length - 1; i++) {
    var phA = PHASES[i], phB = PHASES[i + 1];
    if (!a.phases[phA] || !a.phases[phB]) continue;
    var from = a.phases[phA].centroid;
    var to   = a.phases[phB].centroid;
    annotations.push({
      x: to[0], y: to[1], ax: from[0], ay: from[1],
      xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
      showarrow: true, arrowhead: 3,
      arrowsize: 1.8, arrowwidth: 2.5,
      arrowcolor: 'white', opacity: 0.85,
    });
  }
  // Always show strategy arrow
  annotations.push(strategyArrowAnnotation());

  // No cluster label annotations in employee view — they cover the dots
  var layout = Object.assign({}, baseLayout, { annotations: annotations });
  Plotly.react('map', traces, layout);

  drawRiver(a);
  drawPanel(a, yrCount);
}

// ── Theme river ────────────────────────────────────────────────────
function drawRiver(a) {
  var tr = a.theme_river;
  if (!tr || !tr.years || !tr.years.length) return;

  var traces = tr.series.map(function(s) {
    return {
      type: 'bar',
      name: s.label,
      x: tr.years,
      y: s.counts,
      marker: { color: s.color, opacity: 0.85 },
      hovertemplate: s.label + ': %{y} tickets (%{x})<extra></extra>',
    };
  });

  Plotly.react('river', traces, {
    barmode: 'stack',
    paper_bgcolor: '#0d0d0d', plot_bgcolor: '#111',
    margin: { l: 38, r: 10, t: 8, b: 28 },
    showlegend: false,
    font: { color: '#555', size: 9 },
    xaxis: { color: '#444', tickfont: { size: 9, color: '#666' } },
    yaxis: { color: '#444', tickfont: { size: 9, color: '#666' },
             title: { text: 'tickets/yr', font: { size: 9, color: '#444' } } },
  }, { responsive: true, displayModeBar: false });
}

// ── Strategy context helpers ───────────────────────────────────────
function topThemeNames(arr, n) {
  if (!arr || !arr.length) return '';
  return arr.slice(0, n).map(function(t) {
    return (t && t.label) ? t.label : String(t);
  }).join(' and ');
}

function buildStrategyContext(cos) {
  var earlyFocus  = topThemeNames(S.early_top, 2);
  var recentFocus = topThemeNames(S.recent_top, 2);
  var ctx = 'The project shifted focus from ' + earlyFocus + ' toward ' + recentFocus + '. ';
  if (cos >= 0.7)  ctx += 'This contributor moved strongly in the same direction.';
  else if (cos >= 0.3)  ctx += 'This contributor broadly followed that direction.';
  else if (cos >= -0.3) ctx += 'This contributor took an independent path, working largely outside this main shift.';
  else if (cos >= -0.7) ctx += 'This contributor deepened earlier focus areas as the project moved on.';
  else                  ctx += 'This contributor specialised in areas the project was moving away from.';
  return ctx;
}

// ── Side panel ─────────────────────────────────────────────────────
function drawPanel(a, yrCount) {
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('employee-card').style.display = 'block';

  document.getElementById('c-id').textContent  = 'Employee ' + a.id;
  document.getElementById('c-sub').textContent = a.year_range;
  document.getElementById('c-tc').textContent  = a.ticket_count.toLocaleString();
  document.getElementById('c-yc').textContent  = yrCount;
  document.getElementById('c-al').textContent  = a.alignment != null ? a.alignment + '%' : '—';

  var phDiv = document.getElementById('c-phases');
  phDiv.innerHTML = '';
  PHASES.forEach(function(ph) {
    var info = a.phases[ph];
    if (!info) return;
    var c = document.createElement('div');
    c.className = 'phase-chip';
    c.style.cssText = 'background:' + PHASE_COLORS[ph] + '22;border:1px solid ' +
                       PHASE_COLORS[ph] + ';color:' + PHASE_COLORS[ph];
    c.textContent = ph + ': ' + info.years + ' (' + info.count + ')';
    phDiv.appendChild(c);
  });

  document.getElementById('c-themes').innerHTML =
    (a.top_themes || []).map(function(t) {
      return '<span class="theme-tag">' + t + '</span>';
    }).join('');

  // Prefer UMAP-based alignment; fall back to narrative alignment
  var ua = a.umap_alignment;
  var alignPct  = ua ? ua.umap_align_pct : (a.alignment || 0);
  var alignDesc = ua
    ? (ua.umap_cos >= 0.7  ? 'Moving strongly with the project'
    :  ua.umap_cos >= 0.3  ? 'Generally aligned with project direction'
    :  ua.umap_cos >= -0.3 ? 'Moving independently of the project'
    :  ua.umap_cos >= -0.7 ? 'Moving against the project direction'
    :                        'Moving strongly against the project direction')
    : '';

  var cos = ua ? ua.umap_cos : null;

  // Career trajectory description (text in panel, no arrow on map)
  var dirEl = document.getElementById('c-dir-desc');
  if (cos !== null) {
    var dirText = 'Work shifted from ' + a.year_range.split('-')[0] +
      ' to ' + a.year_range.split('-')[1] + '. ' +
      (cos >= 0.7  ? 'Trajectory closely follows the project direction.'
    :  cos >= 0.3  ? 'Trajectory broadly follows the project direction.'
    :  cos >= -0.3 ? 'Trajectory is largely independent of the project.'
    :  cos >= -0.7 ? 'Focus moved away from where the project was heading.'
    :               'Trajectory is nearly the reverse of the project direction.');
    dirEl.textContent = dirText;
    dirEl.style.display = '';
  } else {
    dirEl.style.display = 'none';
  }

  document.getElementById('c-al').textContent = alignPct + '%';
  document.getElementById('sb-a').textContent = alignPct + '%';
  document.getElementById('c-align-detail').textContent = alignDesc;
  document.getElementById('c-abar').style.width = alignPct + '%';
  document.getElementById('c-abar').style.background =
    alignPct >= 60 ? 'linear-gradient(90deg,#7b68ee,#00cc88)'
  : alignPct >= 40 ? 'linear-gradient(90deg,#7b68ee,#ffa500)'
  :                  'linear-gradient(90deg,#cc4444,#ff6666)';

  // Strategy context paragraph
  var ctxEl = document.getElementById('c-strat-ctx');
  if (cos !== null) {
    ctxEl.textContent = buildStrategyContext(cos);
    ctxEl.style.display = '';
  } else {
    ctxEl.style.display = 'none';
  }

  document.getElementById('c-narr').textContent = a.narrative || 'No narrative available.';
}

// ── Reset ──────────────────────────────────────────────────────────
function resetAll() {
  Plotly.react('map', bgTraces(0.22),
    Object.assign({}, baseLayout,
      { annotations: clusterAnnotations.concat([strategyArrowAnnotation()]) }));
  Plotly.purge('river');
  document.getElementById('stats-bar').style.display = 'none';
  document.getElementById('placeholder').style.display = '';
  document.getElementById('employee-card').style.display = 'none';
  document.getElementById('c-dir-desc').style.display = 'none';
  document.getElementById('c-strat-ctx').style.display = 'none';
}

window.addEventListener('resize', function() {
  Plotly.Plots.resize('map');
  Plotly.Plots.resize('river');
});

// ── Ticket Advisor ─────────────────────────────────────────────────
var ADVISOR_API = '/api/analyze.php';

function toggleAdvisor() {
  var body = document.getElementById('advisor-body');
  var chev = document.getElementById('advisor-chevron');
  if (body.classList.contains('open')) {
    body.classList.remove('open');
    chev.innerHTML = '&#9658;';
  } else {
    body.classList.add('open');
    chev.innerHTML = '&#9660;';
  }
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

var _advisorTraceAdded = false;

function runAnalyze() {
  var text = document.getElementById('ticket-input').value.trim();
  if (!text) return;
  var btn = document.getElementById('analyze-btn');
  btn.disabled = true;
  btn.textContent = 'Analyzing\u2026';
  document.getElementById('advisor-result').innerHTML =
    '<div style="font-size:11px;color:#555;margin-top:8px">Embedding and searching\u2026</div>';

  fetch(ADVISOR_API, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text }),
  })
  .then(function(r) { return r.json(); })
  .then(function(d) {
    if (d.error) throw new Error(d.error);
    showAdvisorResult(d);
  })
  .catch(function(e) {
    document.getElementById('advisor-result').innerHTML =
      '<div class="adv-advice" style="border-left-color:#cc4444;color:#c66;margin-top:8px">' +
      'Error: ' + escHtml(e.message) + '</div>';
  })
  .finally(function() {
    btn.disabled = false;
    btn.textContent = 'Analyze';
  });
}

function showAdvisorResult(d) {
  // Place a star marker on the map at the estimated UMAP position
  if (d.umap_pos && d.umap_pos.x !== undefined) {
    var newTrace = {
      type: 'scatter', mode: 'markers+text',
      x: [d.umap_pos.x], y: [d.umap_pos.y],
      marker: { size: 16, color: '#00ffaa', symbol: 'star',
                line: { color: 'white', width: 2 } },
      text: ['Your ticket'],
      textposition: 'top center',
      textfont: { color: '#00ffaa', size: 10 },
      showlegend: false, hoverinfo: 'text',
      name: 'New ticket',
    };
    if (_advisorTraceAdded) {
      // Replace last trace
      var gd = document.getElementById('map');
      var idx = gd.data.length - 1;
      Plotly.deleteTraces('map', [idx]);
    }
    Plotly.addTraces('map', [newTrace]);
    _advisorTraceAdded = true;
  }

  // Build results HTML
  var html = '<div style="margin-top:10px">';
  if (d.cluster) {
    html += '<div class="sec">Semantic area</div>';
    html += '<div class="theme-tag" style="font-size:11px;margin-bottom:4px">' + escHtml(d.cluster) + '</div>';
  }
  html += '<div class="sec">LLM analysis</div>';
  html += '<div class="adv-advice">' + escHtml(d.advice) + '</div>';

  if (d.similar && d.similar.length) {
    html += '<div class="sec">Similar historical tickets</div>';
    d.similar.forEach(function(t) {
      html += '<div class="adv-sim"><b>' + escHtml(t.key) + '</b> ' +
              escHtml(t.summary) +
              ' <span style="color:#444">(' + Math.round(t.similarity * 100) + '% match)</span></div>';
    });
  }

  if (d.experts && d.experts.length) {
    html += '<div class="sec">Domain experts</div>';
    d.experts.forEach(function(e) {
      html += '<span class="adv-expert-chip">' + escHtml(e.name) + ' (' + e.count + ' similar)</span>';
    });
  }
  html += '</div>';

  document.getElementById('advisor-result').innerHTML = html;
}
</script>
</body>
</html>
"""

out = 'stories_spring/spring_explorer.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Saved {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
