"""回测/选股报告服务：HTML（含图表）/ Excel / PDF 渲染 + 报告历史落盘。

供 routers/reports.py（导出接口）、selection.py / ml.py（回测完成自动存档）复用。
"""
import io
import os
import datetime

from . import db

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def _fmt(v, pct=False):
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v*100:.2f}%" if pct else f"{v:.4f}"


def payload_from_result(result: dict) -> dict:
    """从回测结果 dict 提取报告载荷（前端导出与后端存档共用同一结构）。"""
    return {
        "factorLabel": result.get("factorLabel", "回测"),
        "config": result.get("config") or {},
        "metrics": result.get("metrics") or {},
        "benchmark": result.get("benchmark"),
        "groupSummary": result.get("groupSummary") or [],
        "longShort": result.get("longShort") or [],
        "icSeries": result.get("icSeries") or [],
    }


def render_html(payload: dict, title: str = "回测报告") -> str:
    """自包含 HTML：KPI 卡片 + 表格 + ECharts（CDN）净值/回撤/分组/IC 图。"""
    import json as _json
    m = payload.get("metrics") or {}
    bench = payload.get("benchmark")
    cfg = payload.get("config") or {}
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # 回测概要信息块（起止日 / 历史长度 / 成本 / 候选池 → 解决"报告看不出回测区间"的历史问题）
    start = cfg.get("startDate") or "-"
    end = cfg.get("endDate") or "-"
    hist_val = cfg.get("hist") or "-"
    board_val = cfg.get("board") or "-"
    pool_size = cfg.get("poolSize") or "-"
    n_val = cfg.get("n") or "-"
    cost = f"佣金{_fmt(cfg.get('commissionRate') or 0.00025)} "
    cost += f"印花{_fmt(cfg.get('stampDuty') or 0.001)} "
    cost += f"滑点{_fmt(cfg.get('slippage') or 0.001)}"
    strategy_kind = "ML模型" if cfg.get("modelId") else "技术因子"
    strategy_name = cfg.get("modelId") or payload.get("factorLabel") or "-"
    if strategy_kind == "ML模型":
        strategy_name = cfg.get("modelId") or "-"

    summary = (
        f'<div class="card"><h3>回测概要</h3><table class="summary-table">'
        f'<tr><td>策略来源</td><td>{strategy_kind}</td>'
        f'<td>策略名/模型ID</td><td>{strategy_name}</td></tr>'
        f'<tr><td>回测区间</td><td>{start} ~ {end}</td>'
        f'<td>历史长度(hist)</td><td>{hist_val}</td></tr>'
        f'<tr><td>持有期(n)</td><td>{n_val}</td>'
        f'<td>候选池规模</td><td>{pool_size}</td></tr>'
        f'<tr><td>板块</td><td>{board_val}</td>'
        f'<td>成本</td><td>{cost}</td></tr>'
        f'</table></div>'
    )

    kpi = lambda label, val, pct=False: (
        f'<div class="kpi"><div class="n">{_fmt(val, pct)}</div>'
        f'<div class="l">{label}</div></div>'
    )
    kpis = "".join([
        kpi("累计收益", m.get("cumulativeReturn"), True),
        kpi("年化收益", m.get("annualizedReturn"), True),
        kpi("年化波动", m.get("annualizedVolatility"), True),
        kpi("Sharpe", m.get("sharpe")),
        kpi("Sortino", m.get("sortino")),
        kpi("最大回撤", m.get("maxDrawdown"), True),
        kpi("Calmar", m.get("calmar")),
        kpi("胜率", m.get("winRate"), True),
    ])

    rows = ""
    for g in payload.get("groupSummary") or []:
        rows += (
            f"<tr><td>第{g.get('group')}组</td>"
            f"<td>{_fmt(g.get('avgReturn'), True)}</td>"
            f"<td>{g.get('sample')}</td></tr>"
        )

    bench_html = ""
    if bench:
        bench_html = (
            '<div class="card"><h3>基准对比</h3><table>'
            "<tr><th>指标</th><th>策略</th><th>基准</th></tr>"
            f"<tr><td>累计收益</td><td>{_fmt(m.get('cumulativeReturn'), True)}</td>"
            f"<td>{_fmt(bench.get('cumulativeReturn'), True)}</td></tr>"
            f"<tr><td>年化收益</td><td>{_fmt(m.get('annualizedReturn'), True)}</td>"
            f"<td>{_fmt(bench.get('annualizedReturn'), True)}</td></tr>"
            f"<tr><td>Sharpe</td><td>{_fmt(m.get('sharpe'))}</td>"
            f"<td>{_fmt(bench.get('sharpe'))}</td></tr>"
            f"<tr><td>Alpha</td><td>-</td><td>{_fmt(bench.get('alpha'))}</td></tr>"
            f"<tr><td>Beta</td><td>-</td><td>{_fmt(bench.get('beta'))}</td></tr>"
            "</table></div>"
        )

    ic_stats = ""
    if payload.get("icSeries"):
        ics = [p.get("ic") for p in payload["icSeries"] if p.get("ic") is not None]
        if ics:
            mean_ic = sum(ics) / len(ics)
            pos = sum(1 for v in ics if v > 0) / len(ics)
            ic_stats = (f'<div class="card"><h3>IC 统计</h3><p>'
                        f'平均 IC：{_fmt(mean_ic)} · IC 胜率：{pos*100:.1f}% · 截面数：{len(ics)}</p></div>')

    ls_json = _json.dumps(payload.get("longShort") or [], ensure_ascii=False)
    gs_json = _json.dumps(payload.get("groupSummary") or [], ensure_ascii=False)
    ic_json = _json.dumps(payload.get("icSeries") or [], ensure_ascii=False)
    label = (payload.get("factorLabel") or title).replace("<", "&lt;").replace(">", "&gt;")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>{label} · {title}</title>
<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<style>
body{{font-family:-apple-system,"Segoe UI","PingFang SC",sans-serif;background:#0b1020;color:#e6ebf5;margin:0;padding:32px}}
h1{{font-size:24px;border-left:4px solid #4f8cff;padding-left:10px}}
h3{{color:#cfe0ff;margin-top:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}}
.kpi{{background:#121a30;border:1px solid #23304f;border-radius:10px;padding:14px}}
.kpi .n{{font-size:22px;font-weight:700;color:#4f8cff}}
.kpi .l{{color:#8e9bbd;font-size:12px;margin-top:4px}}
.card{{background:#121a30;border:1px solid #23304f;border-radius:10px;padding:16px;margin:14px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #23304f;padding:8px;text-align:left}}
th{{background:#16213c;color:#cfe0ff}}
.chart{{width:100%;height:340px}}
.sub{{color:#8e9bbd;font-size:13px}}
.summary-table td{{padding:6px 12px; vertical-align:top;}}
.summary-table td:first-child,.summary-table td:nth-child(3){{color:#8e9bbd; font-size:12px; width:100px;}}
@media print{{.chart{{height:300px}} body{{background:#fff;color:#000}} .kpi,.card{{border-color:#ccc;background:#fff}} th{{background:#f0f0f0;color:#000}} h1,h3{{color:#000}}}}
</style></head><body>
	<h1>{label}</h1>
	<div class="sub">生成时间 {now} · 调仓次数 {m.get('rebalanceCount','-')} · 成本率 {_fmt(m.get('costRate'))}</div>
	{summary}
	<div class="grid">{kpis}</div>
{bench_html}
<div class="card"><h3>分组收益</h3><table>
<tr><th>分组</th><th>平均收益</th><th>样本数</th></tr>{rows}</table></div>
{ic_stats}
<div class="card"><h3>多空累计收益与回撤</h3><div id="chartLS" class="chart"></div></div>
<div class="card"><h3>分组平均收益</h3><div id="chartGroup" class="chart"></div></div>
<div class="card"><h3>IC / RankIC 时序列</h3><div id="chartIC" class="chart"></div></div>
<script>
const LS={ls_json}, GS={gs_json}, IC={ic_json};
const dates=LS.map(p=>p.date);
const cum=LS.map(p=>Number((p.cum*100).toFixed(2)));
const top=LS.map(p=>Number(((p.topCum??p.cum)*100).toFixed(2)));
let peak=-Infinity, dd=[];
for(const c of cum){{peak=Math.max(peak,c); dd.push(Number(((c-peak)/100*100).toFixed(2)));}}
echarts.init(document.getElementById('chartLS')).setOption({{
  tooltip:{{trigger:'axis'}}, legend:{{textStyle:{{color:'#8e9bbd'}},top:0}},
  grid:{{left:60,right:40,top:40,bottom:60}},
  xAxis:{{type:'category',data:dates,axisLabel:{{color:'#8e9bbd',rotate:45}}}},
  yAxis:[{{type:'value',axisLabel:{{color:'#8e9bbd',formatter:v=>v+'%'}}}},
         {{type:'value',name:'回撤%',axisLabel:{{color:'#8e9bbd'}},splitLine:{{show:false}}}}],
  series:[
    {{name:'多空累计',type:'line',data:cum,smooth:true,itemStyle:{{color:'#ff4d4f'}},areaStyle:{{color:'rgba(255,77,79,.12)'}}}},
    {{name:'最高组累计',type:'line',data:top,smooth:true,itemStyle:{{color:'#21c08b'}}}},
    {{name:'回撤',type:'line',yAxisIndex:1,data:dd,itemStyle:{{color:'#f39c12'}}}}
  ]
}});
echarts.init(document.getElementById('chartGroup')).setOption({{
  tooltip:{{trigger:'axis'}}, grid:{{left:60,right:30,top:30,bottom:40}},
  xAxis:{{type:'category',data:GS.map(d=>'G'+d.group),axisLabel:{{color:'#8e9bbd'}}}},
  yAxis:{{type:'value',axisLabel:{{color:'#8e9bbd',formatter:v=>(v*100).toFixed(1)+'%'}}}},
  series:[{{type:'bar',data:GS.map(d=>d.avgReturn),itemStyle:{{color:p=>p.value>=0?'#ff4d4f':'#21c08b'}}}}]
}});
echarts.init(document.getElementById('chartIC')).setOption({{
  tooltip:{{trigger:'axis'}}, legend:{{textStyle:{{color:'#8e9bbd'}},top:0}},
  grid:{{left:60,right:30,top:40,bottom:60}},
  xAxis:{{type:'category',data:IC.map(p=>p.date),axisLabel:{{color:'#8e9bbd',rotate:45}}}},
  yAxis:{{type:'value',axisLabel:{{color:'#8e9bbd'}}}},
  series:[
    {{name:'IC',type:'line',data:IC.map(p=>p.ic),showSymbol:false,itemStyle:{{color:'#4f8cff'}}}},
    {{name:'RankIC',type:'line',data:IC.map(p=>p.rankIc),showSymbol:false,itemStyle:{{color:'#6c5ce7'}}}}
  ]
}});
</script>
</body></html>"""


def render_excel(payload: dict) -> bytes:
    """生成 Excel（openpyxl）：绩效指标 / 分组收益 / 多空净值 / IC 序列 多 sheet。"""
    try:
        from openpyxl import Workbook
    except ImportError:
        raise ValueError("服务器未安装 openpyxl，无法导出 Excel")
    wb = Workbook()
    m = payload.get("metrics") or {}

    ws = wb.active
    ws.title = "绩效指标"
    rows = [
        ("因子", payload.get("factorLabel")),
        ("累计收益", m.get("cumulativeReturn")),
        ("年化收益", m.get("annualizedReturn")),
        ("年化波动", m.get("annualizedVolatility")),
        ("Sharpe", m.get("sharpe")),
        ("Sortino", m.get("sortino")),
        ("最大回撤", m.get("maxDrawdown")),
        ("Calmar", m.get("calmar")),
        ("胜率", m.get("winRate")),
        ("调仓次数", m.get("rebalanceCount")),
        ("成本率", m.get("costRate")),
    ]
    for r in rows:
        ws.append(r)

    if payload.get("groupSummary"):
        ws2 = wb.create_sheet("分组收益")
        ws2.append(["分组", "平均收益", "样本数"])
        for g in payload["groupSummary"]:
            ws2.append([g.get("group"), g.get("avgReturn"), g.get("sample")])

    if payload.get("longShort"):
        ws3 = wb.create_sheet("多空净值")
        ws3.append(["日期", "多空收益", "累计收益", "毛收益"])
        for p in payload["longShort"]:
            ws3.append([p.get("date"), p.get("longShort"), p.get("cum"), p.get("gross")])

    if payload.get("icSeries"):
        ws4 = wb.create_sheet("IC序列")
        ws4.append(["日期", "IC", "RankIC", "样本数"])
        for p in payload["icSeries"]:
            ws4.append([p.get("date"), p.get("ic"), p.get("rankIc"), p.get("sample")])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def render_pdf(payload: dict) -> bytes:
    """生成 PDF（reportlab CID 字体 STSong-Light，中文无需外挂字体文件）。

    reportlab 未安装时返回 None，由调用方降级为打印样式 HTML。
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except ImportError:
        return None
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50
    m = payload.get("metrics") or {}
    bench = payload.get("benchmark")

    def line():
        nonlocal y
        c.setFillColorRGB(0.18, 0.22, 0.30)
        c.rect(50, y - 2, w - 100, 0.5, stroke=0, fill=1)
        y -= 14

    def kpi_row(label, val):
        nonlocal y
        if y < 60:
            c.showPage(); y = h - 50
        c.setFont("STSong-Light", 10)
        c.setFillColorRGB(0.15, 0.15, 0.25)
        c.drawString(70, y, label)
        c.setFillColorRGB(0.05, 0.2, 0.6)
        c.drawRightString(w - 70, y, _fmt(val))
        y -= 18

    c.setFont("STSong-Light", 18)
    c.setFillColorRGB(0.05, 0.1, 0.25)
    c.drawString(50, y, payload.get("factorLabel") or "回测报告")
    y -= 22
    c.setFont("STSong-Light", 9)
    c.setFillColorRGB(0.4, 0.4, 0.5)
    c.drawString(50, y, f"生成时间 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 14
    # 回测区间与策略来源（HTML 与 PDF 均含，补齐报告历史缺口）
    cfg = payload.get("config") or {}
    sdate, edate = cfg.get("startDate") or "最远", cfg.get("endDate") or "最新"
    strategy_kind = "ML模型" if cfg.get("modelId") else "技术因子"
    strategy_name = cfg.get("modelId") or payload.get("factorLabel") or "-"
    c.drawString(50, y, f"回测区间 {sdate} ~ {edate}  |  策略来源 {strategy_kind}  |  {strategy_name}")
    y -= 26
    c.setFont("STSong-Light", 13)
    c.drawString(50, y, "绩效指标")
    y -= 20
    for lab, key in (("累计收益", "cumulativeReturn"), ("年化收益", "annualizedReturn"),
                     ("年化波动", "annualizedVolatility"), ("Sharpe", "sharpe"),
                     ("Sortino", "sortino"), ("最大回撤", "maxDrawdown"),
                     ("Calmar", "calmar"), ("胜率", "winRate"),
                     ("调仓次数", "rebalanceCount"), ("成本率", "costRate")):
        kpi_row(lab, m.get(key))
    y -= 6

    if bench:
        if y < 90:
            c.showPage(); y = h - 50
        c.setFont("STSong-Light", 13)
        c.drawString(50, y, "基准对比")
        y -= 20
        for lab, key in (("累计收益", "cumulativeReturn"), ("年化收益", "annualizedReturn"),
                         ("Sharpe", "sharpe"), ("Alpha", "alpha"), ("Beta", "beta")):
            kpi_row(lab, bench.get(key))
        y -= 6

    if payload.get("groupSummary"):
        if y < 120:
            c.showPage(); y = h - 50
        c.setFont("STSong-Light", 13)
        c.drawString(50, y, "分组收益")
        y -= 20
        for g in payload["groupSummary"]:
            kpi_row(f"第{g.get('group')}组", g.get("avgReturn"))
        y -= 6

    if payload.get("icSeries"):
        ics = [p.get("ic") for p in payload["icSeries"] if p.get("ic") is not None]
        if ics:
            mean_ic = sum(ics) / len(ics)
            pos = sum(1 for v in ics if v > 0) / len(ics)
            c.setFont("STSong-Light", 10)
            c.setFillColorRGB(0.15, 0.15, 0.25)
            c.drawString(70, y, f"IC 统计：平均 IC {mean_ic:.4f} · 胜率 {pos*100:.1f}% · 截面 {len(ics)}")
            y -= 24

    if y < 60:
        c.showPage(); y = h - 50
    c.setFont("STSong-Light", 9)
    c.setFillColorRGB(0.5, 0.5, 0.55)
    c.drawString(50, y, "说明：候选池为当前上市标的快照，历史收益可能存在幸存者偏差；"
                        "本报告由量化研究平台自动生成。")
    c.save()
    return buf.getvalue()


def store_backtest_report(result: dict, config: dict | None = None,
                          user_id: int = 0) -> dict | None:
    """回测完成自动存档：生成 HTML 报告文件并登记 backtest_runs 记录。

    返回 run 记录 dict；失败（无 longShort 数据等）返回 None，不阻塞回测主流程。
    """
    try:
        payload = payload_from_result(result)
        if config is not None:
            result["config"] = config
            payload["config"] = config
        if not payload["longShort"] and not payload["icSeries"]:
            return None
        html = render_html(payload)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"report_{ts}.html"
        path = os.path.join(REPORT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        run = db.create_backtest_run(
            None, config or result.get("config") or {},
            result.get("metrics") or {}, report_path=fname,
            user_id=user_id, result=result)
        return run
    except Exception:
        return None


def regenerate_report(run: dict, fmt: str = "html"):
    """从历史 run 的 result 重新生成指定格式报告，返回 (bytes, media_type)。"""
    result = run.get("result") or {}
    if not result:
        return None, None
    payload = payload_from_result(result)
    if fmt == "excel":
        return render_excel(payload), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if fmt == "pdf":
        data = render_pdf(payload)
        if data:
            return data, "application/pdf"
        return None, None
    return render_html(payload), "text/html"

