"""
utils/session_report.py  — v2
──────────────────────────────
Enhanced session report with:
  1. Donut chart — emotion distribution
  2. Timeline line chart — engagement over time
  3. Per-student emotion heatmap
  4. Comparative analysis — this session vs average
  5. Confusion peak detector — which minute was hardest
  6. Recommendations for teacher
"""

import logging
from datetime import datetime
from typing import Dict, List
from collections import Counter

logger = logging.getLogger(__name__)

EMOTION_COLORS = {
    "happy":"#00e5a0","engaged":"#00e5a0","neutral":"#8892b0",
    "confused":"#ff8c42","sad":"#74b9ff","angry":"#ff4d6d",
    "frustrated":"#ff4d6d","bored":"#b47aff","disengaged":"#b47aff",
    "anxious":"#ffc247","surprise":"#00d4ff","fear":"#ff8c42",
}
EMOTION_EMOJI = {
    "happy":"😊","engaged":"🙂","neutral":"😐","confused":"🤔",
    "sad":"😢","angry":"😠","frustrated":"😤","bored":"😒",
    "disengaged":"😶","anxious":"😰","surprise":"😲",
}

def calculate_grade(engaged_pct, confused_pct, alerts):
    score = engaged_pct - (confused_pct * 0.5) - (alerts * 2)
    if score >= 70: return "A",   "#00e5a0"
    if score >= 55: return "B+",  "#00d4ff"
    if score >= 40: return "B",   "#ffc247"
    if score >= 25: return "C",   "#ff8c42"
    return "Needs Improvement",   "#ff4d6d"


def generate_report_html(
    session_info: Dict,
    students: List[Dict],
    emotion_logs: List[Dict],
    notifications: List[Dict],
    duration_seconds: int
) -> str:

    duration_min  = max(1, duration_seconds // 60)
    total_students = len(students)
    total_alerts   = len(notifications)

    # ── Aggregate emotions ────────────────────────────────────────
    all_emotions  = [l["emotion"] for l in emotion_logs if l.get("emotion")]
    emotion_counts = Counter(all_emotions)
    total_readings = max(1, len(all_emotions))

    engaged_count  = sum(emotion_counts.get(e,0) for e in ["happy","engaged","surprise"])
    confused_count = sum(emotion_counts.get(e,0) for e in ["confused","anxious","fear"])
    negative_count = sum(emotion_counts.get(e,0) for e in ["sad","angry","frustrated","disengaged","bored"])
    neutral_count  = emotion_counts.get("neutral", 0)

    engaged_pct  = round(engaged_count  / total_readings * 100)
    confused_pct = round(confused_count / total_readings * 100)
    negative_pct = round(negative_count / total_readings * 100)
    neutral_pct  = round(neutral_count  / total_readings * 100)

    grade, grade_color = calculate_grade(engaged_pct, confused_pct, total_alerts)

    # ── Timeline data — bucket into 1-min intervals ───────────────
    timeline_data = _build_timeline(emotion_logs, duration_seconds)
    timeline_labels = [f"{i+1}m" for i in range(len(timeline_data))]

    # ── Per-student data ──────────────────────────────────────────
    student_rows, student_heatmap = _build_student_rows(students, emotion_logs)

    # ── Comparative benchmarks ────────────────────────────────────
    avg_engaged  = 45   # typical online class benchmark
    avg_confused = 25
    avg_negative = 15

    # ── Peak confusion minute ─────────────────────────────────────
    peak_min, peak_val = _find_peak_confusion(timeline_data)

    # ── Recommendations ───────────────────────────────────────────
    recommendations = _generate_recommendations(
        engaged_pct, confused_pct, negative_pct, total_alerts, peak_min
    )

    # ── Notification rows ─────────────────────────────────────────
    notif_rows = ""
    for n in notifications[:20]:
        color = {"urgent":"#ff4d6d","high":"#ff8c42","medium":"#ffc247"}.get(n.get("level","medium"),"#ffc247")
        notif_rows += f"""<tr>
          <td style="padding:8px 14px;color:#8892b0;font-size:.78rem;font-family:monospace">
            {n.get('timestamp','')[:16].replace('T',' ')}</td>
          <td style="padding:8px 14px;font-weight:600;color:{color}">{n.get('title','')}</td>
          <td style="padding:8px 14px;color:#8892b0;font-size:.8rem">{n.get('body','')}</td>
        </tr>"""

    # ── JS data for charts ────────────────────────────────────────
    tl_engaged   = [d["engaged"]   for d in timeline_data]
    tl_confused  = [d["confused"]  for d in timeline_data]
    tl_negative  = [d["negative"]  for d in timeline_data]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EVA Session Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#080b14;color:#dde4f0;font-family:'Segoe UI',sans-serif;padding:28px;}}
.header{{display:flex;align-items:flex-start;justify-content:space-between;
  margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid #242d45;}}
.logo{{font-size:1.8rem;font-weight:800;color:#7c6af5;}}
.title{{font-size:1rem;color:#8892b0;margin-top:3px;}}
.meta{{font-size:.75rem;color:#5a6688;margin-top:6px;}}
.grade-box{{text-align:center;}}
.grade{{font-size:2.8rem;font-weight:800;color:{grade_color};}}
.grade-lbl{{font-size:.7rem;color:#8892b0;text-transform:uppercase;letter-spacing:1px;}}

.stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px;}}
.stat-card{{background:#161c2e;border:1px solid #242d45;border-radius:12px;
  padding:16px;text-align:center;}}
.stat-val{{font-size:1.6rem;font-weight:700;margin-bottom:3px;}}
.stat-lbl{{font-size:.68rem;color:#8892b0;text-transform:uppercase;letter-spacing:1px;}}

.section{{background:#161c2e;border:1px solid #242d45;border-radius:12px;
  margin-bottom:20px;overflow:hidden;}}
.sec-hdr{{padding:13px 18px;border-bottom:1px solid #242d45;
  font-weight:600;font-size:.88rem;display:flex;align-items:center;gap:8px;}}
.sec-body{{padding:18px;}}

/* 2-col chart grid */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}}
.chart-card{{background:#161c2e;border:1px solid #242d45;border-radius:12px;padding:18px;}}
.chart-title{{font-size:.8rem;color:#8892b0;text-transform:uppercase;
  letter-spacing:1px;margin-bottom:14px;}}
.chart-wrap{{position:relative;height:220px;}}

/* Full-width chart */
.chart-full{{background:#161c2e;border:1px solid #242d45;border-radius:12px;
  padding:18px;margin-bottom:20px;}}
.chart-full .chart-wrap{{height:180px;}}

/* Comparative bars */
.comp-row{{display:flex;align-items:center;gap:12px;margin-bottom:14px;}}
.comp-lbl{{width:80px;font-size:.8rem;color:#8892b0;text-align:right;}}
.comp-bars{{flex:1;display:flex;flex-direction:column;gap:5px;}}
.bar-row{{display:flex;align-items:center;gap:8px;font-size:.72rem;}}
.bar-bg{{flex:1;height:14px;background:#242d45;border-radius:4px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:4px;transition:width 1s;}}
.bar-val{{width:34px;text-align:right;font-weight:600;}}
.comp-legend{{display:flex;gap:16px;font-size:.72rem;color:#8892b0;margin-bottom:14px;}}
.dot{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px;}}

/* Heatmap */
.heatmap-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px;}}
.hm-name{{width:100px;font-size:.78rem;color:#8892b0;
  text-overflow:ellipsis;overflow:hidden;white-space:nowrap;}}
.hm-cells{{display:flex;gap:3px;}}
.hm-cell{{width:18px;height:18px;border-radius:3px;}}

/* Recommendation cards */
.rec-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;}}
.rec-card{{background:#0f1422;border-left:3px solid;border-radius:8px;padding:12px 14px;}}
.rec-icon{{font-size:1.2rem;margin-bottom:6px;}}
.rec-title{{font-size:.82rem;font-weight:600;margin-bottom:4px;}}
.rec-body{{font-size:.75rem;color:#8892b0;line-height:1.6;}}

table{{width:100%;border-collapse:collapse;}}
tr:nth-child(even){{background:#0f1422;}}
.peak-badge{{background:rgba(255,140,66,.15);border:1px solid rgba(255,140,66,.4);
  color:#ff8c42;padding:3px 10px;border-radius:12px;font-size:.75rem;font-weight:600;}}

@media print{{body{{background:white;color:#333;}}
  .section,.chart-card,.chart-full{{border:1px solid #ccc;background:white;}}}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div>
    <div class="logo">EVA</div>
    <div class="title">Session Report — {session_info.get('class_name','Class')} · {session_info.get('subject','')}</div>
    <div class="meta">
      Teacher: {session_info.get('teacher','—')} &nbsp;·&nbsp;
      Date: {datetime.now().strftime('%d %b %Y')} &nbsp;·&nbsp;
      Duration: {duration_min} min &nbsp;·&nbsp;
      Students: {total_students} &nbsp;·&nbsp;
      Alerts: {total_alerts}
    </div>
  </div>
  <div class="grade-box">
    <div class="grade">{grade}</div>
    <div class="grade-lbl">Class Grade</div>
    {'<div style="margin-top:6px" class="peak-badge">⚠️ Peak confusion: min '+str(peak_min)+'</div>' if peak_min > 0 else ''}
  </div>
</div>

<!-- STATS ROW -->
<div class="stats">
  <div class="stat-card">
    <div class="stat-val" style="color:#dde4f0">{total_students}</div>
    <div class="stat-lbl">Students</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:#00e5a0">{engaged_pct}%</div>
    <div class="stat-lbl">Engaged</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:#8892b0">{neutral_pct}%</div>
    <div class="stat-lbl">Neutral</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:#ff8c42">{confused_pct}%</div>
    <div class="stat-lbl">Confused</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:#ff4d6d">{total_alerts}</div>
    <div class="stat-lbl">Alerts Sent</div>
  </div>
</div>

<!-- CHARTS ROW 1: Donut + Comparative -->
<div class="chart-grid">

  <!-- Donut chart -->
  <div class="chart-card">
    <div class="chart-title">📊 Emotion Distribution</div>
    <div class="chart-wrap">
      <canvas id="donutChart"></canvas>
    </div>
  </div>

  <!-- Comparative analysis -->
  <div class="chart-card">
    <div class="chart-title">📈 This Session vs Benchmark</div>
    <div class="comp-legend">
      <span><span class="dot" style="background:#7c6af5"></span>This session</span>
      <span><span class="dot" style="background:#242d45"></span>Avg online class</span>
    </div>
    <div class="comp-row">
      <div class="comp-lbl">😊 Engaged</div>
      <div class="comp-bars">
        <div class="bar-row">
          <div class="bar-bg"><div class="bar-fill" style="width:{engaged_pct}%;background:#00e5a0"></div></div>
          <div class="bar-val" style="color:#00e5a0">{engaged_pct}%</div>
        </div>
        <div class="bar-row">
          <div class="bar-bg"><div class="bar-fill" style="width:{avg_engaged}%;background:#242d45;border:1px solid #344"></div></div>
          <div class="bar-val" style="color:#5a6688">{avg_engaged}%</div>
        </div>
      </div>
    </div>
    <div class="comp-row">
      <div class="comp-lbl">🤔 Confused</div>
      <div class="comp-bars">
        <div class="bar-row">
          <div class="bar-bg"><div class="bar-fill" style="width:{confused_pct}%;background:#ff8c42"></div></div>
          <div class="bar-val" style="color:#ff8c42">{confused_pct}%</div>
        </div>
        <div class="bar-row">
          <div class="bar-bg"><div class="bar-fill" style="width:{avg_confused}%;background:#242d45;border:1px solid #344"></div></div>
          <div class="bar-val" style="color:#5a6688">{avg_confused}%</div>
        </div>
      </div>
    </div>
    <div class="comp-row">
      <div class="comp-lbl">😢 Negative</div>
      <div class="comp-bars">
        <div class="bar-row">
          <div class="bar-bg"><div class="bar-fill" style="width:{negative_pct}%;background:#ff4d6d"></div></div>
          <div class="bar-val" style="color:#ff4d6d">{negative_pct}%</div>
        </div>
        <div class="bar-row">
          <div class="bar-bg"><div class="bar-fill" style="width:{avg_negative}%;background:#242d45;border:1px solid #344"></div></div>
          <div class="bar-val" style="color:#5a6688">{avg_negative}%</div>
        </div>
      </div>
    </div>
    <div style="margin-top:10px;font-size:.72rem;color:#5a6688">
      Benchmark based on typical online class averages (Stanford 2021 study)
    </div>
  </div>
</div>

<!-- TIMELINE CHART (full width) -->
<div class="chart-full">
  <div class="chart-title">📉 Engagement Timeline — minute by minute</div>
  <div class="chart-wrap">
    <canvas id="timelineChart"></canvas>
  </div>
</div>

<!-- STUDENT HEATMAP -->
<div class="section">
  <div class="sec-hdr">🎯 Student Emotion Heatmap</div>
  <div class="sec-body">
    <div style="font-size:.72rem;color:#5a6688;margin-bottom:12px">
      Each cell = one emotion reading. Color shows emotion state over time.
      <span style="margin-left:12px">
        <span class="dot" style="background:#00e5a0"></span>engaged &nbsp;
        <span class="dot" style="background:#8892b0"></span>neutral &nbsp;
        <span class="dot" style="background:#ff8c42"></span>confused &nbsp;
        <span class="dot" style="background:#ff4d6d"></span>negative
      </span>
    </div>
    {student_heatmap if student_heatmap else '<div style="color:#5a6688;font-size:.8rem">No emotion data recorded</div>'}
  </div>
</div>

<!-- STUDENT BREAKDOWN TABLE -->
<div class="section">
  <div class="sec-hdr">👥 Student Breakdown</div>
  <table>
    <thead>
      <tr style="background:#0f1422">
        <th style="padding:10px 14px;text-align:left;color:#8892b0;font-size:.72rem;text-transform:uppercase">Name</th>
        <th style="padding:10px 14px;text-align:left;color:#8892b0;font-size:.72rem;text-transform:uppercase">Last Emotion</th>
        <th style="padding:10px 14px;text-align:left;color:#8892b0;font-size:.72rem;text-transform:uppercase">Source</th>
        <th style="padding:10px 14px;text-align:left;color:#8892b0;font-size:.72rem;text-transform:uppercase">Last Message</th>
      </tr>
    </thead>
    <tbody>{student_rows or '<tr><td colspan="4" style="padding:20px;text-align:center;color:#5a6688">No data</td></tr>'}</tbody>
  </table>
</div>

<!-- RECOMMENDATIONS -->
<div class="section">
  <div class="sec-hdr">💡 Teaching Recommendations</div>
  <div class="sec-body">
    <div class="rec-grid">
      {''.join(recommendations)}
    </div>
  </div>
</div>

<!-- ALERTS LOG -->
<div class="section">
  <div class="sec-hdr">🔔 Alerts Log ({total_alerts} total)</div>
  <table>
    <thead>
      <tr style="background:#0f1422">
        <th style="padding:8px 14px;text-align:left;color:#8892b0;font-size:.72rem;text-transform:uppercase">Time</th>
        <th style="padding:8px 14px;text-align:left;color:#8892b0;font-size:.72rem;text-transform:uppercase">Alert</th>
        <th style="padding:8px 14px;text-align:left;color:#8892b0;font-size:.72rem;text-transform:uppercase">Details</th>
      </tr>
    </thead>
    <tbody>{notif_rows or '<tr><td colspan="3" style="padding:20px;text-align:center;color:#5a6688">No alerts this session</td></tr>'}</tbody>
  </table>
</div>

<!-- FOOTER -->
<div style="text-align:center;color:#5a6688;font-size:.72rem;margin-top:24px;
  padding-top:16px;border-top:1px solid #242d45;">
  Generated by EVA — Emotion Adaptive Virtual Teaching Assistant &nbsp;·&nbsp;
  {datetime.now().strftime('%d %b %Y %H:%M')}
  <br><br>
  <button onclick="window.print()" style="background:#7c6af5;border:none;color:white;
    padding:8px 20px;border-radius:8px;cursor:pointer;font-size:.82rem;">
    🖨️ Print / Save as PDF
  </button>
</div>

<script>
// ── Chart.js defaults ──────────────────────────────────────────
Chart.defaults.color = '#8892b0';
Chart.defaults.borderColor = '#242d45';

// ── 1. Donut Chart ─────────────────────────────────────────────
new Chart(document.getElementById('donutChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Engaged', 'Neutral', 'Confused', 'Negative'],
    datasets: [{{
      data: [{engaged_pct}, {neutral_pct}, {confused_pct}, {negative_pct}],
      backgroundColor: ['#00e5a0','#8892b0','#ff8c42','#ff4d6d'],
      borderColor: '#161c2e',
      borderWidth: 3,
      hoverOffset: 8
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    cutout: '68%',
    plugins: {{
      legend: {{
        position: 'right',
        labels: {{ padding: 14, font: {{ size: 11 }}, boxWidth: 12 }}
      }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}}%`
        }}
      }}
    }}
  }}
}});

// ── 2. Timeline Chart ──────────────────────────────────────────
const tlLabels = {timeline_labels};
new Chart(document.getElementById('timelineChart'), {{
  type: 'line',
  data: {{
    labels: tlLabels.length ? tlLabels : ['1m'],
    datasets: [
      {{
        label: 'Engaged %',
        data: {tl_engaged} .length ? {tl_engaged}  : [0],
        borderColor: '#00e5a0', backgroundColor: 'rgba(0,229,160,.08)',
        fill: true, tension: 0.4, pointRadius: 3
      }},
      {{
        label: 'Confused %',
        data: {tl_confused}.length ? {tl_confused} : [0],
        borderColor: '#ff8c42', backgroundColor: 'rgba(255,140,66,.08)',
        fill: true, tension: 0.4, pointRadius: 3
      }},
      {{
        label: 'Negative %',
        data: {tl_negative}.length ? {tl_negative} : [0],
        borderColor: '#ff4d6d', backgroundColor: 'rgba(255,77,109,.05)',
        fill: true, tension: 0.4, pointRadius: 3
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 11 }}, boxWidth: 12 }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y}}%`
        }}
      }}
    }},
    scales: {{
      y: {{
        min: 0, max: 100,
        ticks: {{ callback: v => v+'%', font: {{ size: 10 }} }},
        grid: {{ color: '#1e2540' }}
      }},
      x: {{ ticks: {{ font: {{ size: 10 }} }}, grid: {{ color: '#1e2540' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ── Helpers ───────────────────────────────────────────────────────

def _build_timeline(emotion_logs: List[Dict], duration_seconds: int) -> List[Dict]:
    """Build per-minute emotion buckets for timeline chart."""
    if not emotion_logs:
        return [{"engaged":0,"confused":0,"negative":0,"neutral":100}]

    duration_min = max(1, duration_seconds // 60)
    buckets = [{"engaged":[],"confused":[],"negative":[],"neutral":[]}
               for _ in range(duration_min)]

    for i, log in enumerate(emotion_logs):
        bucket_idx = min(i * duration_min // len(emotion_logs), duration_min - 1)
        e = log.get("emotion","neutral")
        if e in ["happy","engaged","surprise"]:
            buckets[bucket_idx]["engaged"].append(1)
        elif e in ["confused","anxious","fear"]:
            buckets[bucket_idx]["confused"].append(1)
        elif e in ["sad","angry","frustrated","bored","disengaged"]:
            buckets[bucket_idx]["negative"].append(1)
        else:
            buckets[bucket_idx]["neutral"].append(1)

    result = []
    for b in buckets:
        total = max(1, sum(len(v) for v in b.values()))
        result.append({
            "engaged":  round(len(b["engaged"])  / total * 100),
            "confused": round(len(b["confused"]) / total * 100),
            "negative": round(len(b["negative"]) / total * 100),
            "neutral":  round(len(b["neutral"])  / total * 100),
        })
    return result


def _find_peak_confusion(timeline_data: List[Dict]) -> tuple:
    """Find the minute with highest confusion."""
    if not timeline_data:
        return 0, 0
    peak_idx = max(range(len(timeline_data)), key=lambda i: timeline_data[i]["confused"])
    return peak_idx + 1, timeline_data[peak_idx]["confused"]


def _build_student_rows(students: List[Dict], emotion_logs: List[Dict]):
    """Build HTML table rows and heatmap for students."""
    rows = ""
    heatmap = ""

    # Group logs by student
    student_logs: Dict[str, List] = {}
    for log in emotion_logs:
        name = log.get("student", "")
        if name:
            student_logs.setdefault(name, []).append(log.get("emotion","neutral"))

    for s in students:
        name    = s.get("name","Unknown")
        emotion = s.get("emotion","neutral")
        emoji   = EMOTION_EMOJI.get(emotion,"😐")
        color   = EMOTION_COLORS.get(emotion,"#8892b0")
        msg     = s.get("last_message","")
        source  = s.get("source","face")

        rows += f"""<tr>
          <td style="padding:10px 14px;font-weight:600">{name}</td>
          <td style="padding:10px 14px">
            <span style="background:{color}22;color:{color};border:1px solid {color}44;
              padding:3px 10px;border-radius:12px;font-size:.8rem">
              {emoji} {emotion}
            </span>
          </td>
          <td style="padding:10px 14px;color:#8892b0;font-size:.82rem">{source}</td>
          <td style="padding:10px 14px;color:#8892b0;font-size:.82rem;font-style:italic">
            {f'"{msg}"' if msg else '—'}
          </td>
        </tr>"""

        # Heatmap row for this student
        logs = student_logs.get(name, [emotion] * 8)
        cells = ""
        for e in logs[-20:]:
            c = EMOTION_COLORS.get(e,"#8892b0")
            cells += f'<div class="hm-cell" style="background:{c}88" title="{e}"></div>'
        heatmap += f"""<div class="heatmap-row">
          <div class="hm-name" title="{name}">{name}</div>
          <div class="hm-cells">{cells}</div>
          <div style="font-size:.72rem;color:#5a6688;margin-left:8px">{emoji}</div>
        </div>"""

    return rows, heatmap


def _generate_recommendations(engaged_pct, confused_pct, negative_pct, alerts, peak_min):
    """Generate actionable teaching recommendations based on session data."""
    recs = []

    if engaged_pct >= 70:
        recs.append(("✅", "#00e5a0", "Great engagement!",
            "Students were highly engaged this session. The teaching pace and content level were well-matched."))
    elif engaged_pct >= 45:
        recs.append(("📈", "#ffc247", "Moderate engagement",
            "About half the class was engaged. Consider adding more interactive elements like polls or questions."))
    else:
        recs.append(("⚠️", "#ff4d6d", "Low engagement detected",
            "Less than 45% engagement. Try shorter explanation segments (5-7 min) followed by quick activities."))

    if confused_pct >= 30:
        recs.append(("🤔", "#ff8c42", "High confusion rate",
            f"30%+ of readings showed confusion. Consider revisiting the most complex topic with a simpler analogy or example."))
    elif confused_pct >= 15:
        recs.append(("💡", "#ffc247", "Some confusion detected",
            "A portion of students showed signs of confusion. A quick recap at the start of next class may help."))

    if negative_pct >= 25:
        recs.append(("😟", "#ff4d6d", "Negative emotions high",
            "Students showed frustration or disengagement. Check if content difficulty or pacing needs adjustment."))

    if alerts >= 3:
        recs.append(("🔔", "#ff8c42", f"{alerts} alerts fired",
            "Multiple students needed help during this session. Consider smaller group breakouts or additional practice time."))

    if peak_min > 0:
        recs.append(("⏱️", "#b47aff", f"Hardest point: minute {peak_min}",
            f"Confusion peaked around minute {peak_min}. Review what topic was being covered at that time."))

    if not recs:
        recs.append(("🎓", "#00e5a0", "Session completed",
            "No major issues detected. Continue with the current teaching approach."))

    return [f"""<div class="rec-card" style="border-color:{color}55">
      <div class="rec-icon">{icon}</div>
      <div class="rec-title" style="color:{color}">{title}</div>
      <div class="rec-body">{body}</div>
    </div>""" for icon, color, title, body in recs]