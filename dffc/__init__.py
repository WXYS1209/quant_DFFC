"""
DFFC 模块

提供中国金融市场数据获取功能，包括基金、股票等资产类型
集成vectorbt框架，提供统一的数据接口

模块结构:
- data_provider: 数据提供者模块
- strategies: 量化交易策略模块  
- holt_winters: Holt-Winters时间序列分析模块
- fund_data: 基金数据处理模块
"""

# 版本信息
__version__ = "0.2.0"
__author__ = "DFFC Team"
__description__ = "DFFC Quant Module based on vectorbt"
__license__ = "MIT"

# 导入工具函数
from dffc._utils import *

# 导入数据提供者
from dffc.data_provider import *

# 导入Holt-Winters模块
from dffc.holt_winters import *

# 导入核心功能
from dffc.fund_data import *
from dffc.stock_data import *

# 导入策略模块 
from dffc.strategies import *

# 导入回测模块
from dffc.backtesting import *

# 导入估值工具
from dffc.estimate_utils import *

# 定义公开API
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__description__",
    "__license__",
    
    # 核心模块
    "fund_data",
    "data_provider",
    "strategies", 
    "holt_winters",
    "backtesting",
    "estimate_utils",
]