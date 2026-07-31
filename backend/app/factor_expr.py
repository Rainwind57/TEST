"""因子表达式引擎（P1-6）：让用户用字符串表达式定义截面因子。

旧版仅 composite_score 固定规格加权（key/weight/direction），
无法表达 momentum - volatility*0.5、max(rsi, 30) 等任意非线性组合。
本引擎用 AST 白名单求值，只允许算术运算 + 列引用 +安全函数，杜绝任意代码执行。

用法：
    score = evaluate_expression("momentum - volatility * 0.5", rows)
    # rows: [{code, momentum, volatility, rsi, ...}, ...]
    # 返回每行打分列表，NaN 视为 0
"""
import ast
import math
import operator as op

import numpy as np

# AST 节点白名单：只允许这些构造，其余（属性访问/调用任意函数/import）一律拒
# 注意 ast.Add/Sub/Mult 等既作 BinOp.op 也可能与 ast.Subscript 冲突，需一并放行 op 节点
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant,
    ast.Load,
    # 函数调用：只允许白名单函数（见下方 Call 校验）
    ast.Call,
    # 运算符 op 节点（walk 会访问到，必须放行）
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.UAdd, ast.USub, ast.Not,
)

# 显式拒这些节点（即使 walk 遇到也拒）
_FORBIDDEN_NODES = (
    ast.Attribute, ast.Subscript, ast.Import, ast.ImportFrom,
    ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.Assign, ast.AugAssign, ast.AnnAssign,
)

# 二元运算白名单
_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod,
}
# 一元运算白名单
_UNARY_OPS = {
    ast.UAdd: op.pos, ast.USub: op.neg, ast.Not: op.not_,
}

# 允许调用的安全函数白名单（numpy 标量/向量均兼容）
_SAFE_FUNCS = {
    "abs": np.abs, "max": np.maximum, "min": np.minimum,
    "log": np.log, "exp": np.exp, "sqrt": np.sqrt,
    "sign": np.sign, "clip": np.clip,
    "mean": np.mean, "std": np.std, "sum": np.sum,
    "rank": lambda a: _rank(a),
    "zscore": lambda a: _zscore(a),
}


def _rank(arr):
    """截面排名归一化到 [0,1]。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return a
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1)
    return (ranks - 1) / max(1, a.size - 1)


def _zscore(arr):
    """截面 z-score，常量列返回 0。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return a
    m, s = np.nanmean(a), np.nanstd(a)
    if s == 0:
        return np.zeros_like(a)
    return (a - m) / s


def validate_expression(expr: str) -> tuple[bool, str]:
    """校验表达式安全性，返回 (ok, error_msg)。"""
    if not expr or not expr.strip():
        return False, "表达式为空"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    for node in ast.walk(tree):
        # 黑名单：属性访问/下标/import/推导式/赋值等一律拒（防任意代码执行）
        if isinstance(node, _FORBIDDEN_NODES):
            return False, f"不允许的语法: {type(node).__name__}"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False, "只允许调用白名单函数"
            if node.func.id not in _SAFE_FUNCS:
                return False, f"不允许的函数: {node.func.id}"
    return True, ""


def evaluate_expression(expr: str, rows: list[dict]) -> list[float]:
    """对 rows 截面求值表达式，返回每行打分。

    列引用从 row 字段取值；NaN 视为 0（避免传染整个表达式）。
    表达式非法时抛 ValueError。
    """
    ok, err = validate_expression(expr)
    if not ok:
        raise ValueError(err)
    if not rows:
        return []
    tree = ast.parse(expr, mode="eval")
    # 收集表达式引用的列名，统一转 numpy 数组（向量运算）
    col_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                 and n.id not in _SAFE_FUNCS}
    columns: dict[str, np.ndarray] = {}
    for name in col_names:
        vals = []
        for r in rows:
            v = r.get(name)
            try:
                vals.append(float(v) if v is not None else 0.0)
            except (TypeError, ValueError):
                vals.append(0.0)
        columns[name] = np.array(vals, dtype=np.float64)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in _SAFE_FUNCS:
                return _SAFE_FUNCS[node.id]
            return columns.get(node.id, np.zeros(len(rows)))
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.Call):
            fn = _SAFE_FUNCS[node.func.id]
            args = [_eval(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"不支持的节点: {type(node).__name__}")

    result = _eval(tree.body)
    arr = np.asarray(result, dtype=np.float64)
    if arr.ndim == 0:  # 常量表达式
        arr = np.full(len(rows), float(arr))
    # NaN → 0
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr.tolist()
