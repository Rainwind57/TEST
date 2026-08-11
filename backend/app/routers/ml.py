"""机器学习路由：构建数据集 → 时序 CV 评估 → 训练并落盘模型。"""
import asyncio
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel

from .. import ml, jobs
from .auth import require_user_id

router = APIRouter(prefix="/api/ml", tags=["ml"])


class EvalBody(BaseModel):
    board: str = "all"
    boards: list[str] | None = None  # 多板块 OR 组合（如 ["sh_main","gem"]），优先于 board
    poolSize: int = 100
    n: int = 5
    hist: int = ml.ML_DEFAULT_HIST  # 默认 1024：覆盖内置长周期因子（momentum120/dist_52w_high），开箱即用
    modelType: str = "gbdt"
    nSplits: int = 5
    gap: int = 5
    useSnapshot: bool = False  # 追加 pe/pb/turniture 快照特征（含前视风险，探索用）
    nTrials: int = 30  # ml-optimize 用（Optuna 试验数）
    assetClass: str = "a-share"  # a-share | future（期货主力连续合约池）
    startDate: str | None = None  # 训练集样本起始日（YYYY-MM-DD，含），限定分时段训练
    endDate: str | None = None    # 训练集样本结束日（YYYY-MM-DD，含）
    selectedFactors: list[str] | None = None  # 指定因子子集训练（key 列表），不传=全部


@router.post("/evaluate")
async def evaluate(body: EvalBody, uid: int = Depends(require_user_id)):
    """同步评估（小数据集）。大数据集建议走 /api/jobs 异步提交。"""
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist,
                                         use_snapshot=body.useSnapshot, asset_class=body.assetClass,
                                         start_date=body.startDate, end_date=body.endDate,
                                         boards=body.boards,
                                         selected_factors=body.selectedFactors)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, ml.evaluate_dataset, dataset, body.modelType, body.nSplits, body.gap)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if dataset.get("snapshotWarning"):
        result["snapshotWarning"] = dataset["snapshotWarning"]
    return result

@router.post("/train")
async def train(body: EvalBody, uid: int = Depends(require_user_id)):
    """构建数据集 + 训练最终模型并落盘。"""
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist,
                                         use_snapshot=body.useSnapshot, asset_class=body.assetClass,
                                         start_date=body.startDate, end_date=body.endDate,
                                         boards=body.boards,
                                         selected_factors=body.selectedFactors)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    # CPU 密集同步函数放线程池，避免阻塞事件循环（旧版训练期间全站请求卡死）
    loop = asyncio.get_running_loop()
    try:
        eval_result = await loop.run_in_executor(None, ml.evaluate_dataset, dataset, body.modelType, body.nSplits, body.gap)
    except ValueError as e:
        raise HTTPException(422, str(e))
    meta = await loop.run_in_executor(None, ml.train_final_model, dataset, body.modelType)
    result = {"model": meta, "evaluation": eval_result}
    if dataset.get("snapshotWarning"):
        result["snapshotWarning"] = dataset["snapshotWarning"]
    return result


@router.get("/models")
def list_models(uid: int = Depends(require_user_id)):
    return ml.list_models()


class OptimizeMlBody(BaseModel):
    board: str = "all"
    boards: list[str] | None = None  # 多板块 OR 组合，优先于 board
    poolSize: int = 100
    n: int = 5
    hist: int = ml.ML_DEFAULT_HIST  # 默认 1024：覆盖内置长周期因子，寻优开箱即用
    modelType: str = "lightgbm"  # 默认启用已装的 lightgbm（旧版仅 gbdt）
    nSplits: int = 5
    gap: int = 5
    nTrials: int = 30
    useSnapshot: bool = False
    assetClass: str = "a-share"  # a-share | future
    startDate: str | None = None  # 训练集起始日（分时段训练）
    endDate: str | None = None    # 训练集结束日
    selectedFactors: list[str] | None = None  # 可选：指定因子子集


@router.post("/optimize")
async def optimize(body: OptimizeMlBody, uid: int = Depends(require_user_id)):
    """ML 超参寻优（Optuna + Walk-Forward OOS Sharpe）。

    旧版 Optuna 仅接因子回测（optimize.optimize_backtest），ML _build_model
    硬编码超参、无法自动寻优；此端点打通 ML 调参闭环，并落盘实验记录。
    大数据集建议走 /api/jobs（kind=ml-optimize）。
    """
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist,
                                         use_snapshot=body.useSnapshot, asset_class=body.assetClass,
                                         start_date=body.startDate, end_date=body.endDate,
                                         boards=body.boards,
                                         selected_factors=body.selectedFactors)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, ml.optimize_model, dataset, body.modelType, body.nSplits, body.gap, body.nTrials)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if dataset.get("snapshotWarning"):
        result["snapshotWarning"] = dataset["snapshotWarning"]
    return result


@router.post("/models/import")
async def import_model(file: UploadFile = File(...), uid: int = Depends(require_user_id)):
    """导入外部训练好的模型文件（joblib bundle：model + feature_names [+ preprocess]）。

    落盘到 ml_models/ 并登记元数据，之后即可与平台内模型同等用于打分/回测/盯盘。
    """
    data = await file.read()
    try:
        meta = ml.import_model_file(file.filename or "model.joblib", data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return meta


@router.get("/models/import-template")
def get_import_template(uid: int = Depends(require_user_id)):
    """外部模型导入引导：平台特征清单 + 示例打包代码（P8 降低导入门槛）。"""
    return ml.model_import_template()


class ManualModelBody(BaseModel):
    name: str = ""
    featureWeights: dict[str, float] = {}
    threshold: float | None = None
    rule: str = ""


@router.post("/models/manual")
def create_manual_model(body: ManualModelBody, uid: int = Depends(require_user_id)):
    """创建"人造/手动模型"：手工指定因子权重 + 阈值（可选规则说明），
    落盘为与自动训练产物同构的 bundle，可立即用于打分/回测/盯盘调度。"""
    try:
        meta = ml.create_manual_model(body.name, body.featureWeights, body.threshold, body.rule)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return meta


@router.get("/manual/features")
def manual_features(uid: int = Depends(require_user_id)):
    """手动模型可用的因子清单（与平台 build_dataset 技术因子同构）。"""
    return ml.manual_feature_options()


@router.delete("/models/{mid}")
def delete_model(mid: str, uid: int = Depends(require_user_id)):
    if not ml.delete_model(mid):
        raise HTTPException(404, "模型不存在")
    return {"ok": True}


class ModelAdjustBody(BaseModel):
    featureWeights: dict[str, float] | None = None   # 特征权重覆盖（key=featureName, value=weight）
    threshold: float | None = None                    # 预测阈值偏移
    saveArtifact: bool = False                        # 调整配置落盘，供打分/回测引用


@router.get("/models/{mid}/params")
def get_model_params(mid: str, uid: int = Depends(require_user_id)):
    """查看已落盘模型的参数（特征重要性 + 超参 + 特征名），供人工调参参考。

    读取失败时返回结构化 500 而非让 worker 崩溃（旧版 load_model_meta 无条件
    joblib.load 整包，跨 lightgbm 版本反序列化可能段错误 → 前端 network error）。
    """
    try:
        meta = ml.load_model_meta(mid)
    except Exception as e:
        raise HTTPException(500, f"模型元数据读取失败（模型文件可能损坏或版本不兼容）: {e}")
    if not meta:
        raise HTTPException(404, "模型不存在")
    return meta


@router.post("/models/{mid}/adjust")
def adjust_model(mid: str, body: ModelAdjustBody, uid: int = Depends(require_user_id)):
    """人工调整模型输出：覆盖特征权重 或 偏移预测阈值。

    调整配置落盘为中间结果（kind=ml_adjust），打分层/回测接口传 adjustId 即可生效，
    形成"调参→回测验证→再调"闭环，打通 ML→选股/回测。
    对树模型（GBDT/LightGBM）说明调参的实际语义：权重是对输入特征做线性缩放，
    经分裂路径间接改变预测，并非改写模型内部系数。
    """
    try:
        meta = ml.load_model_meta(mid)
    except Exception as e:
        raise HTTPException(500, f"模型元数据读取失败（模型文件可能损坏或版本不兼容）: {e}")
    if not meta:
        raise HTTPException(404, "模型不存在")
    model_type = meta.get("modelType", "gbdt")
    tree_like = model_type in ("gbdt", "lightgbm", "gradient_boosting", "lgbm")
    adjust_cfg = {"modelId": mid, "featureNames": meta.get("featureNames", [])}
    if body.featureWeights is not None:
        valid_features = set(meta.get("featureNames", []))
        fw = {k: v for k, v in body.featureWeights.items() if k in valid_features}
        if fw:
            adjust_cfg["featureWeights"] = fw
    if body.threshold is not None:
        adjust_cfg["threshold"] = body.threshold
    effect_note = (
        f"模型类型 {model_type}。特征权重会对输入特征做线性缩放后再预测（非改写模型内部系数），"
        f"对树模型效果有限；如需真正「调整系数」，建议用「人造模型」（人工加权线性模型）或重训练。"
        if tree_like else
        f"模型类型 {model_type}（线性/人工加权），特征权重即为最终系数，调整直接生效。"
    )
    result = {"modelId": mid, "adjustConfig": adjust_cfg, "originalMeta": meta,
              "effectNote": effect_note}
    if body.saveArtifact:
        from .. import artifacts
        meta2 = artifacts.save_artifact("ml_adjust", adjust_cfg,
                                        name=f"调参-{mid}")
        result["artifact"] = meta2
        result["adjustId"] = meta2["id"]
    return result


def _load_adjust(adjustId: str | None, adjust: dict | None) -> dict | None:
    """从 artifact 或直传 dict 解析调参配置。"""
    if adjust:
        return {"modelId": adjust.get("modelId", ""), "featureNames": adjust.get("featureNames", []),
                "featureWeights": adjust.get("featureWeights") or {},
                "threshold": adjust.get("threshold")}
    if adjustId:
        from .. import artifacts
        rec = artifacts.load_artifact(adjustId)
        if not rec:
            raise HTTPException(404, f"调参配置不存在: {adjustId}")
        cfg = rec.get("payload", {})
        return {"modelId": cfg.get("modelId", ""), "featureNames": cfg.get("featureNames", []),
                "featureWeights": cfg.get("featureWeights") or {},
                "threshold": cfg.get("threshold")}
    return None


class ScoreBody(BaseModel):
    modelId: str
    board: str = "all"
    boards: list[str] | None = None  # 多板块 OR 组合，优先于 board
    poolSize: int = 100
    saveArtifact: bool = False   # 打分结果落盘为中间结果（选股结果 codes 供下一环节复用）
    adjustId: str | None = None  # 调参配置 artifact id（/models/{mid}/adjust 产出）
    adjust: dict | None = None   # 或直接传 {featureWeights, threshold}
    assetClass: str = "a-share"  # a-share | future


@router.post("/score")
async def score(body: ScoreBody, uid: int = Depends(require_user_id)):
    """用落盘模型对候选池最新截面打分（打通 ML→选股：结果可加入自选/买入模拟盘）。

    adjustId/adjust 传调参配置时应用人工权重/阈值。
    """
    try:
        adjust = _load_adjust(body.adjustId, body.adjust)
        rows = await ml.score_latest(body.modelId, body.board, body.poolSize,
                                     adjust=adjust, asset_class=body.assetClass,
                                     boards=body.boards)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"打分失败: {e}")
    if body.saveArtifact:
        from .. import artifacts
        meta = artifacts.save_artifact("ml_score", {
            "modelId": body.modelId,
            "adjust": adjust,
            "codes": [r["code"] for r in rows],
            "rows": rows,
            "config": body.model_dump(),
        }, name=f"ML打分-{body.modelId}")
        return {"rows": rows, "artifact": meta}
    return rows


class MLBacktestBody(BaseModel):
    modelId: str
    board: str = "all"
    boards: list[str] | None = None  # 多板块 OR 组合，优先于 board
    poolSize: int = 150
    groups: int = 3
    n: int = 3
    hist: int = 180
    commissionRate: float = 0.00025
    stampDuty: float = 0.001
    slippage: float = 0.001
    benchmark: str = "none"
    applyCost: bool = True
    adjustId: str | None = None  # 调参配置 artifact id
    adjust: dict | None = None   # 或直接传 {featureWeights, threshold}
    assetClass: str = "a-share"  # a-share | future
    startDate: str | None = None  # 验证区间起始日（YYYY-MM-DD，含），分时段验证
    endDate: str | None = None    # 验证区间结束日（YYYY-MM-DD，含）


@router.post("/backtest")
async def ml_backtest(body: MLBacktestBody, uid: int = Depends(require_user_id)):
    """ML 信号分层回测（打通 ML→回测，响应结构与 /api/select/backtest 一致，前端图表零成本复用）。

    adjustId/adjust 传调参配置时应用人工权重，验证调参后的模型表现。
    startDate/endDate 限定验证回测的调仓日区间（分时段验证）。
    """
    try:
        adjust = _load_adjust(body.adjustId, body.adjust)
        config = body.model_dump()
        # 注入模型元数据供报告渲染模型类型/超参/特征重要性
        try:
            meta = ml.load_model_meta(body.modelId)
        except Exception:
            meta = None
        if meta:
            config["_modelMeta"] = {
                "modelType": meta.get("modelType", "gbdt"),
                "featureNames": meta.get("featureNames") or [],
                "featureImportance": meta.get("featureImportance") or [],
                "bestParams": meta.get("bestParams"),
            }
        result = await ml.backtest_model(
            body.modelId, body.board, body.poolSize, body.groups, body.n, body.hist,
            body.commissionRate, body.stampDuty, body.slippage, body.benchmark, body.applyCost,
            adjust=adjust, asset_class=body.assetClass,
            start_date=body.startDate, end_date=body.endDate,
            config=config,
            boards=body.boards,
        )
        # 回测存档带 user_id
        try:
            from .. import reporting
            reporting.store_backtest_report(result, config=config, user_id=uid)
        except Exception:
            pass
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"ML 回测失败: {e}")
