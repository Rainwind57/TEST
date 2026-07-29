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
