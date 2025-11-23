"""
策略模块 (Strategies)

提供量化交易策略的实现，包括基础策略类和具体策略实现

模块结构:
- base: 策略基类和通用再平衡策略框架

主要类:
- Strategy: 抽象策略基类
- ReallocationStrategy: 通用再平衡策略基类
"""

from dffc.strategies.base import (
    Strategy,
    ReallocationStrategy
)

__all__ = [
    # 基础策略类
    "Strategy",
    "ReallocationStrategy"
]

# 策略模块元信息
__module_name__ = "strategies"
__module_description__ = "量化交易策略实现模块"
__supported_strategies__ = [
    "基础策略框架",
    "通用再平衡策略"
]