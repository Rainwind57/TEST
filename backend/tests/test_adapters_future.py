"""adapters.py 期货代码映射单元测试。

覆盖：
- 股指/国债 → CFF_ 前缀（支持具体月份合约）
- 商品 → 主力连续合约（RB0/AU0/M0 格式）
- 已带前缀的代码幂等
"""
from app import adapters


def test_index_future_keeps_month():
    """股指期货保留月份：IF2608 → CFF_IF2608。"""
    assert adapters._future_secid("IF2608") == "CFF_IF2608"


def test_index_future_cff_families():
    for fam in ["IF", "IH", "IC", "IM", "T", "TF", "TS"]:
        assert adapters._future_secid(f"{fam}2609").startswith("CFF_")


def test_commodity_maps_to_continuous():
    """商品期货自动映射主力连续：rb2610 → RB0。"""
    assert adapters._future_secid("rb2610") == "RB0"
    assert adapters._future_secid("au2612") == "AU0"
    assert adapters._future_secid("m2609") == "M0"


def test_continuous_code_idempotent():
    """已是连续合约代码的保持原样：RB0 → RB0。"""
    assert adapters._future_secid("RB0") == "RB0"


def test_prefixed_code_stripped():
    """带前缀的代码先剥前缀再映射。"""
    assert adapters._future_secid("CFF_IF2608") == "CFF_IF2608"
    assert adapters._future_secid("NF_rb2610") == "RB0"


def test_is_number_helper():
    assert adapters._is_number("4557.0") is True
    assert adapters._is_number("螺纹钢连续") is False
    assert adapters._is_number("") is False
