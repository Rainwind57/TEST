"""数据扩展路由：资金流向、北向资金、财务指标。"""
from fastapi import APIRouter, HTTPException

from .. import adapters

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/money-flow/{code}")
async def money_flow(code: str):
    try:
        return await adapters.fetch_money_flow(code)
    except Exception as e:
        raise HTTPException(502, f"资金流向获取失败: {e}")


@router.get("/money-flow-trend/{code}")
async def money_flow_trend(code: str, days: int = 10):
    try:
        return await adapters.fetch_money_flow_trend(code, max(1, min(days, 60)))
    except Exception as e:
        raise HTTPException(502, f"资金流向趋势获取失败: {e}")


@router.get("/north-flow")
async def north_flow(days: int = 30):
    try:
        return await adapters.fetch_north_flow_trend(max(1, min(days, 120)))
    except Exception as e:
        raise HTTPException(502, f"北向资金获取失败: {e}")


@router.get("/finance/{code}")
async def finance_summary(code: str):
    try:
        return await adapters.fetch_finance_summary(code)
    except Exception as e:
        raise HTTPException(502, f"财务指标获取失败: {e}")


@router.get("/sectors")
async def sector_map():
    """全市场股票→申万一级行业映射。"""
    try:
        return await adapters.fetch_sector_map()
    except Exception as e:
        raise HTTPException(502, f"行业分类获取失败: {e}")


@router.get("/macro/{indicator}")
async def macro_indicator(indicator: str):
    """宏观指标：CPI/PPI/PMI/M2。"""
    try:
        result = await adapters.fetch_macro_indicator(indicator)
        if not result:
            raise HTTPException(404, f"指标 {indicator} 不存在或数据不可用")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"宏观数据获取失败: {e}")


@router.get("/index-constituents/{index_code}")
async def index_constituents(index_code: str):
    """指数成分股列表（如 sh000300=沪深300）。"""
    try:
        codes = await adapters.fetch_index_constituents(index_code)
        return {"index": index_code, "count": len(codes), "codes": codes}
    except Exception as e:
        raise HTTPException(502, f"指数成分获取失败: {e}")


@router.get("/futures/quotes")
async def future_quotes(codes: str):
    """期货实时行情，codes 逗号分隔（如 IF2406,rb2410,au2412）。"""
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    if not code_list:
        raise HTTPException(400, "codes 不能为空")
    try:
        return await adapters.fetch_future_quotes(code_list)
    except Exception as e:
        raise HTTPException(502, f"期货行情获取失败: {e}")


@router.get("/futures/kline/{code}")
async def future_kline(code: str, days: int = 150):
    """期货日K线。"""
    try:
        rows = await adapters.fetch_future_kline(code, max(60, min(days, 500)))
        return {"code": code, "count": len(rows), "rows": rows}
    except Exception as e:
        raise HTTPException(502, f"期货K线获取失败: {e}")
