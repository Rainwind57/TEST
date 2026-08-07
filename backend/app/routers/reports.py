"""回测报告导出：HTML（含图表）/ Excel / PDF + 报告历史存档管理 + 选股报告。"""
import datetime
import os
from fastapi import APIRouter, HTTPException, Depends, Response
from pydantic import BaseModel

from .. import db, reporting
from .auth import require_user_id

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ExportBody(BaseModel):
    format: str  # "html" | "excel" | "pdf"
    factorLabel: str
    config: dict = {}
    metrics: dict = {}
    benchmark: dict | None = None
    groupSummary: list[dict] = []
    longShort: list[dict] = []
    icSeries: list[dict] = []


def _payload_from_body(body: ExportBody) -> dict:
    return {
        "factorLabel": body.factorLabel,
        "config": body.config,
        "metrics": body.metrics,
        "benchmark": body.benchmark,
        "groupSummary": body.groupSummary,
        "longShort": body.longShort,
        "icSeries": body.icSeries,
    }


@router.post("/backtest")
def export_backtest(body: ExportBody):
    """回测报告导出：html（含图表）/ excel / pdf。"""
    payload = _payload_from_body(body)
    fmt = body.format
    if fmt == "html":
        content = reporting.render_html(payload)
        return Response(content=content, media_type="text/html",
                        headers={"Content-Disposition": f"attachment; filename=backtest_{int(datetime.datetime.now().timestamp())}.html"})
    if fmt == "excel":
        try:
            data = reporting.render_excel(payload)
        except ValueError as e:
            raise HTTPException(500, str(e))
        return Response(content=data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f"attachment; filename=backtest_{int(datetime.datetime.now().timestamp())}.xlsx"})
    if fmt == "pdf":
        data = reporting.render_pdf(payload)
        if data is None:
            # reportlab 缺失降级：返回打印样式 HTML（浏览器打印→PDF）
            content = reporting.render_html(payload)
            return Response(content=content, media_type="text/html",
                            headers={"Content-Disposition": f"attachment; filename=backtest_{int(datetime.datetime.now().timestamp())}.html"})
        return Response(content=data, media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=backtest_{int(datetime.datetime.now().timestamp())}.pdf"})
    raise HTTPException(400, "format 必须为 html / excel / pdf")


# ---------------- 报告历史（回测完成自动存档） ----------------

@router.get("/runs")
def list_report_runs(limit: int = 50, uid: int = Depends(require_user_id)):
    return db.list_backtest_runs(max(1, min(limit, 200)), user_id=uid)


@router.get("/runs/{run_id}")
def download_report(run_id: int, uid: int = Depends(require_user_id)):
    """下载历史报告文件（HTML），优先用存档文件，缺失时由 result 重新渲染。"""
    run = _get_run(run_id, uid)
    path = run.get("report_path")
    if path and os.path.exists(os.path.join(reporting.REPORT_DIR, path)):
        with open(os.path.join(reporting.REPORT_DIR, path), "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content=content, media_type="text/html",
                        headers={"Content-Disposition": f"attachment; filename={path}"})
    result = run.get("result") or {}
    if result:
        payload = reporting.payload_from_result(result)
        return Response(content=reporting.render_html(payload), media_type="text/html",
                        headers={"Content-Disposition": f"attachment; filename=report_{run_id}.html"})
    raise HTTPException(404, "该报告无存档数据，无法下载")


@router.post("/runs/{run_id}/regenerate")
def regenerate_report(run_id: int, fmt: str = "html", uid: int = Depends(require_user_id)):
    """从历史 run 存储的 result 一键重生成 html/excel/pdf 报告。"""
    run = _get_run(run_id, uid)
    data, media_type = reporting.regenerate_report(run, fmt)
    if data is None:
        if fmt == "pdf":
            data = reporting.render_html(reporting.payload_from_result(run.get("result") or {}))
            media_type = "text/html"
        else:
            raise HTTPException(400, "该报告无存档数据，无法重新生成")
    ext = "html" if "html" in media_type else "xlsx" if "spreadsheetml" in media_type else "pdf"
    return Response(content=data, media_type=media_type,
                    headers={"Content-Disposition": f"attachment; filename=report_{run_id}.{ext}"})


@router.delete("/runs/{run_id}")
def delete_report(run_id: int, uid: int = Depends(require_user_id)):
    run = _get_run(run_id, uid)
    path = run.get("report_path")
    if path:
        try:
            os.remove(os.path.join(reporting.REPORT_DIR, path))
        except OSError:
            pass
    if not db.delete_backtest_run(run_id, user_id=uid):
        raise HTTPException(404, "报告记录不存在")
    return {"ok": True}


def _get_run(run_id: int, uid: int) -> dict:
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM backtest_runs WHERE id = ? AND (user_id = ? OR user_id = 0)",
        (run_id, uid)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "报告记录不存在")
    return db._parse_backtest_run(row)


# ---------------- 选股结果报告 ----------------

class SelectReportBody(BaseModel):
    format: str = "html"       # html | excel | pdf
    board: str = ""
    poolSize: int = 0
    topN: int = 0
    rows: list[dict] = []
    config: dict = {}


@router.post("/select")
def export_select_report(body: SelectReportBody, uid: int = Depends(require_user_id)):
    """选股结果报告：候选池参数 + TopN 榜单（代码/名称/得分/关键因子值）。"""
    rows = body.rows or []
    fmt = body.format
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    label = f"选股报告 · {body.board} · 前{body.topN or len(rows)}"

    if fmt == "excel":
        try:
            from openpyxl import Workbook
            import io
        except ImportError:
            raise HTTPException(500, "服务器未安装 openpyxl，无法导出 Excel")
        wb = Workbook()
        ws = wb.active
        ws.title = "选股结果"
        ws.append(["排名", "代码", "名称", "现价", "涨跌幅", "综合得分"])
        for r in rows:
            ws.append([r.get("rank"), r.get("code"), r.get("name"),
                       r.get("price"), r.get("pctChg"), r.get("score")])
        buf = io.BytesIO()
        wb.save(buf)
        return Response(content=buf.getvalue(),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f"attachment; filename=select_{int(datetime.datetime.now().timestamp())}.xlsx"})

    rows_html = ""
    for r in rows:
        rows_html += (f"<tr><td>{r.get('rank')}</td><td>{r.get('code')}</td><td>{r.get('name')}</td>"
                      f"<td>{r.get('price')}</td><td>{r.get('pctChg')}</td><td>{r.get('score')}</td></tr>")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>{label}</title>
<style>
body{{font-family:-apple-system,"Segoe UI","PingFang SC",sans-serif;background:#0b1020;color:#e6ebf5;margin:0;padding:32px}}
h1{{font-size:24px;border-left:4px solid #4f8cff;padding-left:10px}}
.card{{background:#121a30;border:1px solid #23304f;border-radius:10px;padding:16px;margin:14px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #23304f;padding:8px;text-align:left}}
th{{background:#16213c;color:#cfe0ff}}
.sub{{color:#8e9bbd;font-size:13px}}
</style></head><body>
<h1>{label}</h1>
<div class="sub">生成时间 {now} · 候选池 {body.poolSize} · 榜单 {len(rows)} 只</div>
<div class="card"><table><tr><th>排名</th><th>代码</th><th>名称</th><th>现价</th><th>涨跌幅</th><th>综合得分</th></tr>{rows_html}</table></div>
</body></html>"""
    if fmt == "html":
        return Response(content=html, media_type="text/html",
                        headers={"Content-Disposition": f"attachment; filename=select_{int(datetime.datetime.now().timestamp())}.html"})
    if fmt == "pdf":
        data = reporting.render_pdf({
            "factorLabel": label,
            "metrics": {"rebalanceCount": len(rows)},
            "benchmark": None, "groupSummary": [], "longShort": [], "icSeries": [],
        })
        if data is None:
            return Response(content=html, media_type="text/html",
                            headers={"Content-Disposition": f"attachment; filename=select_{int(datetime.datetime.now().timestamp())}.html"})
        return Response(content=data, media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=select_{int(datetime.datetime.now().timestamp())}.pdf"})
    raise HTTPException(400, "format 必须为 html / excel / pdf")
