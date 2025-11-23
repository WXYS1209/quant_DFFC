"""
Holt-Winters 时间序列分析模块

提供 Holt-Winters 三次指数平滑算法的实现和优化功能

模块结构:
- _holt_winters: 核心Holt-Winters算法实现
- _optimization: 参数优化功能

主要功能:
- HWDP: Holt-Winters双参数分析
- HW_ETS: 指数三次平滑算法  
- 参数优化: 自动寻找最优alpha、beta、gamma参数
- 多资产并行处理
"""

from dffc.holt_winters._holt_winters import (
    HW,
    HWD,
    HWDP,
    # 导入其他可用的类和函数
)

from dffc.holt_winters._optimization import (
    process_hw_opt,
    # 导入其他优化相关函数
)

__all__ = [
    # 核心算法
    "HW", 
    "HWD",
    "HWDP",
    
    # 优化功能
    "process_hw_opt",
]

# 模块元信息
__module_name__ = "holt_winters"
__module_description__ = "Holt-Winters时间序列分析和预测模块"
__algorithms__ = [
    "Holt-Winters双参数分析 (HWDP)",
    "指数三次平滑 (HW)", 
    "参数自动优化"
]