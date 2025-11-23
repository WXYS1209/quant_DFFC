from dffc.backtesting.base import (
    BackTest,
    ReallocationBackTest
)

__all__ = [
    # 回测类
    "BackTest",
    "ReallocationBackTest"
    
]

# 策略模块元信息
__module_name__ = "backtest"
__module_description__ = "量化交易回测模块"
__supported_backtests__ = [
    "回测框架",
    "再平衡回测框架"
]