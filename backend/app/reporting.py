"""回测/选股报告服务：HTML（含图表）/ Excel / PDF 渲染 + 报告历史落盘。

供 routers/reports.py（导出接口）、selection.py / ml.py（回测完成自动存档）复用。
"""
import base64
import io
import os
import datetime
import logging

from . import db

logger = logging.getLogger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_ECHARTS_PATH = os.path.join(_STATIC_DIR, "echarts.min.js")


def _echarts_script_tag() -> str:
    """返回 ECharts 加载脚本标签。

    优先从 backend/app/static/echarts.min.js 内联 base64（完全离线、自包含），
    若文件不存在则降级为多 CDN 链（国内→国际）。
    """
    if os.path.exists(_ECHARTS_PATH):
        with open(_ECHARTS_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<script src="data:text/javascript;base64,{b64}"></script>'

    # 降级：多 CDN 容错链（cdn.bootcdn.net 在国内较稳，再备 unpkg / cdnjs）
    return (
        '<script>'
        '(function(){'
        'var u=["https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js",'
        '"https://unpkg.com/echarts@5.5.0/dist/echarts.min.js",'
        '"https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js"],'
        'i=0;function t(){if(i>=u.length){'
        'document.getElementById("chart-fallback").style.display="block";return}'
        'var s=document.createElement("script");s.src=u[i++];'
        's.onerror=t;document.head.appendChild(s)}t()'
        '})()'
        '</script>'
    )

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
    cfg = result.get("config") or {}
    return {
        "factorLabel": result.get("factorLabel", "回测"),
        "config": cfg,
        "metrics": result.get("metrics") or {},
        "benchmark": result.get("benchmark"),
        "groupSummary": result.get("groupSummary") or [],
        "longShort": result.get("longShort") or [],
        "icSeries": result.get("icSeries") or [],
        "survivorshipBiasWarning": result.get("survivorshipBiasWarning", ""),
        "snapshotWarning": result.get("snapshotWarning", ""),
        "longOnly": result.get("longOnly", True),
        "longOnlyNote": result.get("longOnlyNote", ""),
        "longShortNote": result.get("longShortNote", ""),
        # 模型详情：config 中携带的模型 meta（routers/ml.py 回测时注入）
        "_modelMeta": cfg.get("_modelMeta") if cfg.get("modelId") else None,
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

    # 警告横幅：前视偏差 / 生存偏差 / 多空不可实盘
    warnings = []
    sw = payload.get("snapshotWarning", "")
    if sw:
        warnings.append(f'<div class="warn-banner" style="border-color:#e6a817;background:#1a1410">⚠️ 前视偏差警告：{sw}</div>')
    sb = payload.get("survivorshipBiasWarning", "")
    if sb:
        warnings.append(f'<div class="warn-banner">⚠️ 生存偏差：{sb}</div>')
    lo = payload.get("longOnly", True)
    if lo:
        note = payload.get("longOnlyNote", "")
        if note:
            warnings.append(f'<div class="info-banner">📌 {note}</div>')
    else:
        note = payload.get("longShortNote", "")
        if note:
            warnings.append(f'<div class="warn-banner" style="border-color:#e6a817;background:#1a1410">⚠️ {note}</div>')

    # 因子说明块（技术因子回测无模型详情时，补充因子定义/方向/逻辑）
    factor_html = ""
    if not model_meta and payload.get("factorLabel"):
        factor_key = payload.get("factorLabel", "")
        try:
            from .factors import FACTORS
            fdef = FACTORS.get(factor_key)
        except Exception:
            fdef = None
        if not fdef:
            try:
                from .ml import FACTORS as ML_FACTORS
                fdef = ML_FACTORS.get(factor_key)
            except Exception:
                fdef = None
        if fdef:
            factor_html = (
                f'<div class="card"><h3>因子说明</h3><table class="summary-table">'
                f'<tr><td>因子名称</td><td>{fdef.get("label", factor_key)}</td>'
                f'<td>方向</td><td>{fdef.get("direction", "-")}</td></tr>'
                f'<tr><td>分组</td><td>{fdef.get("group", "-")}</td>'
                f'<td>窗口</td><td>{fdef.get("window", "-")}</td></tr>'
                f'</table></div>'
            )

    # 模型详情块
    model_meta = payload.get("_modelMeta")
    model_html = ""
    if model_meta:
        model_type = model_meta.get("modelType", "gbdt")
        n_features = len(model_meta.get("featureNames") or [])
        bp = model_meta.get("bestParams")
        hp_str = ""
        if bp:
            parts = []
            if "n_estimators" in bp:
                parts.append(f"n_estimators={bp['n_estimators']}")
            if "max_depth" in bp:
                parts.append(f"max_depth={bp['max_depth']}")
            if "learning_rate" in bp:
                parts.append(f"learning_rate={bp['learning_rate']}")
            hp_str = " · ".join(parts)
        top_features = (model_meta.get("featureImportance") or [])[:5]
        ft_rows = "".join(
            f"<tr><td>{f['feature']}</td><td>{f['importance']:.4f}</td></tr>"
            for f in top_features
        )
        model_html = (
            f'<div class="card"><h3>模型详情</h3><table class="summary-table">'
            f'<tr><td>模型类型</td><td>{model_type}</td>'
            f'<td>特征数</td><td>{n_features}</td></tr>'
            + (f'<tr><td>超参数</td><td colspan="3">{hp_str}</td></tr>' if hp_str else "") +
            f'</table>'
            f'<h4 style="margin-top:12px">Top 特征重要性</h4>'
            f'<table><tr><th>特征</th><th>重要性</th></tr>{ft_rows}</table></div>'
        )

    # 调仓方式文字
    groups_val = cfg.get("groups") or "-"
    cost_note = ""
    if cfg.get("applyCost"):
        cost_note = (
            "交易成本（佣金+印花税+滑点）已计入。高频调仓（n≤3日）时成本侵蚀显著，"
            "建议关注减成本后指标（如净Sharpe与毛Sharpe的差异）。"
        )
    else:
        cost_note = "⚠️ 未计入交易成本，实际收益需扣除佣金/印花税/滑点。"
    rebalance_desc = (
        f"分层回测 · 每 {n_val} 交易日按因子值全市场排序分 {groups_val} 组 · "
        + ("做多Top组（仅多头可实盘，空头需融券）" if lo else f"做多第1组 / 做空第{groups_val}组（研究用多空组合）")
        + (f" · T+1开盘入场 · {cost_note}")
    )

    warnings_html = "\n".join(warnings) if warnings else ""

    summary = (
        f'{warnings_html}'
        f'{factor_html}'
        f'{model_html}'
        f'<div class="card"><h3>回测概要</h3><table class="summary-table">'
        f'<tr><td>调仓方式</td><td colspan="3">{rebalance_desc}</td></tr>'
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

    # IC 统计
    ic_stats = ""
    ics: list[float] = []
    mean_ic = 0.0
    pos = 0.0
    if payload.get("icSeries"):
        ics = [p.get("ic") for p in payload["icSeries"] if p.get("ic") is not None]
        if ics:
            mean_ic = sum(ics) / len(ics)
            pos = sum(1 for v in ics if v > 0) / len(ics)
            ic_stats = (f'<div class="card"><h3>IC 统计</h3><p>'
                        f'平均 IC：{_fmt(mean_ic)} · IC 胜率：{pos*100:.1f}% · 截面数：{len(ics)}</p></div>')

    # 结果解读
    sharpe = m.get("sharpe") or 0
    mdd = m.get("maxDrawdown") or 0
    wr = m.get("winRate") or 0
    ic_val = mean_ic
    ic_pos = pos

    sharpe_grade = "优秀（≥1.5）" if sharpe >= 1.5 else "良好（1.0~1.5）" if sharpe >= 1.0 else "一般（0.5~1.0）" if sharpe >= 0.5 else "偏低（<0.5）"
    mdd_grade = "较大（<-30%）" if mdd < -0.3 else "中等（-20%~-30%）" if mdd < -0.2 else "可控（>-20%）"
    ic_grade = "有效（均值>0.03）" if ic_val > 0.03 else "中等（0.01~0.03）" if ic_val > 0.01 else "偏弱（<0.01）"
    conclusion = []
    if sharpe >= 1.0 and mdd > -0.25:
        conclusion.append("策略整体风险调整后收益表现良好，回撤可控，适合作为组合基础仓位。")
    elif sharpe >= 0.5:
        conclusion.append("策略有一定选股能力但回撤偏高，建议结合仓位管理或止损机制使用。")
    else:
        conclusion.append("策略信号偏弱，建议优化因子组合或扩大候选池后再评估。")
    if ic_val > 0.02:
        conclusion.append("IC 均值表明因子对截面收益有区分度，可配合其他因子构建多因子模型。")
    if len(ics) < 20 and len(ics) > 0:
        conclusion.append(f"调仓次数仅 {len(ics)} 次，样本偏少，绩效指标统计显著性不足。")

    interp_html = (
        f'<div class="card"><h3>结果解读</h3>'
        f'<p style="line-height:1.8;color:#cfe0ff">Sharpe {sharpe_grade} · '
        f'最大回撤 {mdd_grade} · 胜率 {wr*100:.1f}% · '
        f'因子IC {ic_grade}。{" ".join(conclusion)}</p>'
        f'</div>'
    )

    ls_json = _json.dumps(payload.get("longShort") or [], ensure_ascii=False)
    gs_json = _json.dumps(payload.get("groupSummary") or [], ensure_ascii=False)
    ic_json = _json.dumps(payload.get("icSeries") or [], ensure_ascii=False)
    label = (payload.get("factorLabel") or title).replace("<", "&lt;").replace(">", "&gt;")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>{label} · {title}</title>
{_echarts_script_tag()}
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
.warn-banner{{background:#1a1410;border:1px solid #e6604d;border-radius:8px;padding:12px 16px;margin:12px 0;color:#ffa39e;font-size:13px;line-height:1.6}}
.info-banner{{background:#101a25;border:1px solid #4f8cff;border-radius:8px;padding:12px 16px;margin:12px 0;color:#a0c4ff;font-size:13px;line-height:1.6}}
@media print{{.chart{{height:300px}} body{{background:#fff;color:#000}} .kpi,.card{{border-color:#ccc;background:#fff}} th{{background:#f0f0f0;color:#000}} h1,h3{{color:#000}}}}
	</style></head><body>
		<div id="chart-fallback" style="display:none;background:#2a1a1a;border:1px solid #ff4d4f;border-radius:8px;padding:16px;margin:16px 0;color:#ffa39e;font-size:13px">⚠️ 图表加载失败，请检查网络连接后刷新页面，或导出 Excel/PDF 格式查看数据</div>
		<h1>{label}</h1>
	<div class="sub">生成时间 {now} · 调仓次数 {m.get('rebalanceCount','-')} · 成本率 {_fmt(m.get('costRate'))}</div>
	{summary}
	<div class="grid">{kpis}</div>
{bench_html}
<div class="card"><h3>分组收益</h3><table>
<tr><th>分组</th><th>平均收益</th><th>样本数</th></tr>{rows}</table></div>
{ic_stats}
{interp_html}
<div class="card"><h3>多空累计收益与回撤</h3><p class="sub" style="margin:4px 0 8px">蓝线为多空组合累计收益率（%），红色阴影为回撤。长期右上→策略有效。</p><div id="chartLS" class="chart"></div></div>
<div class="card"><h3>分组平均收益</h3><p class="sub" style="margin:4px 0 8px">柱状图为各分组的平均持仓期收益，分组1→N 单调→因子区分度好。</p><div id="chartGroup" class="chart"></div></div>
<div class="card"><h3>IC / RankIC 时序列</h3><p class="sub" style="margin:4px 0 8px">IC（皮尔逊）/ RankIC（斯皮尔曼）时序。>0 天数多→因子预测方向稳，|IC|大→区分度强。</p><div id="chartIC" class="chart"></div></div>
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
    cfg = payload.get("config") or {}

    ws = wb.active
    ws.title = "绩效指标"
    rows = [
        ("因子/模型", payload.get("factorLabel")),
        ("板块", cfg.get("board", "-")),
        ("持有期(n)", cfg.get("n", "-")),
        ("历史长度(hist)", cfg.get("hist", "-")),
        ("分组数", cfg.get("groups", "-")),
        ("候选池规模", cfg.get("poolSize", "-")),
        ("回测起始", cfg.get("startDate", "-")),
        ("回测结束", cfg.get("endDate", "-")),
        ("佣金/印花/滑点", f"{cfg.get('commissionRate',0.00025)}/{cfg.get('stampDuty',0.001)}/{cfg.get('slippage',0.001)}"),
        ("", ""),
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
    except Exception as e:
        logger.warning("回测报告自动存档失败: %s", e)
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

