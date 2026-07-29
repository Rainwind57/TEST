"""回测报告导出：HTML（自包含）与 Excel（openpyxl）。"""
import io
import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ExportBody(BaseModel):
    format: str  # "html" | "excel"
    factorLabel: str
    config: dict
    metrics: dict
    benchmark: dict | None = None
    groupSummary: list[dict] = []
    longShort: list[dict] = []
    icSeries: list[dict] = []


def _fmt(v, pct=False):
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v*100:.2f}%" if pct else f"{v:.4f}"


@router.post("/backtest")
def export_backtest(body: ExportBody):
    if body.format == "html":
        return _export_html(body)
    if body.format == "excel":
        return _export_excel(body)
    raise HTTPException(400, "format 必须为 html 或 excel")


def _export_html(body: ExportBody) -> Response:
    m = body.metrics or {}
    bench = body.benchmark
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

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
    for g in body.groupSummary:
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

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>回测报告 · {body.factorLabel}</title>
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
.sub{{color:#8e9bbd;font-size:13px}}
</style></head><body>
<h1>分层回测报告 · {body.factorLabel}</h1>
<div class="sub">生成时间 {now} · 调仓次数 {m.get('rebalanceCount','-')} · 成本率 {_fmt(m.get('costRate'))}</div>
<div class="grid">{kpis}</div>
{bench_html}
<div class="card"><h3>分组收益</h3><table>
<tr><th>分组</th><th>平均收益</th><th>样本数</th></tr>{rows}</table></div>
</body></html>"""
    return Response(content=html, media_type="text/html",
                    headers={"Content-Disposition": f"attachment; filename=backtest_{int(datetime.datetime.now().timestamp())}.html"})


def _export_excel(body: ExportBody) -> Response:
    try:
        from openpyxl import Workbook
    except ImportError:
        raise HTTPException(500, "服务器未安装 openpyxl，无法导出 Excel")

    wb = Workbook()
    m = body.metrics or {}

    ws = wb.active
    ws.title = "绩效指标"
    rows = [
        ("因子", body.factorLabel),
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

    if body.groupSummary:
        ws2 = wb.create_sheet("分组收益")
        ws2.append(["分组", "平均收益", "样本数"])
        for g in body.groupSummary:
            ws2.append([g.get("group"), g.get("avgReturn"), g.get("sample")])

    if body.longShort:
        ws3 = wb.create_sheet("多空净值")
        ws3.append(["日期", "多空收益", "累计收益", "毛收益"])
        for p in body.longShort:
            ws3.append([p.get("date"), p.get("longShort"), p.get("cum"), p.get("gross")])

    if body.icSeries:
        ws4 = wb.create_sheet("IC序列")
        ws4.append(["日期", "IC", "RankIC", "样本数"])
        for p in body.icSeries:
            ws4.append([p.get("date"), p.get("ic"), p.get("rankIc"), p.get("sample")])

    buf = io.BytesIO()
    wb.save(buf)
    data = buf.getvalue()
    return Response(
        content=data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=backtest_{int(datetime.datetime.now().timestamp())}.xlsx"},
    )
