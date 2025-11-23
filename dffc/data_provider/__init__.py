"""
数据提供者模块 (Data Provider)

提供统一的数据提供者抽象层，支持多种资产类型和数据源

模块结构:
- base: 数据提供者基类和配置
- eastmoney_provider: 东方财富数据提供者
- stock_net_value_crawler: 股票净值爬虫

主要功能:
- 基金数据获取
- 股票数据获取  
- 统一的数据接口
- 多数据源支持
"""

from dffc.data_provider.base import (
    DataProvider, 
    DataProviderConfig, 
    BS4DataProvider
)

from dffc.data_provider.eastmoney_provider import (
    EastMoneyFundProvider, 
    EastMoneyStockProvider
)

from dffc.data_provider.fund_est_val_crawler import FundEstimateProvider

# 尝试导入爬虫模块
try:
    from dffc.data_provider.stock_net_value_crawler import *
except ImportError:
    pass  # 如果依赖不满足，忽略导入错误

__all__ = [
    # 基础类
    'DataProviderConfig', 
    'DataProvider',
    'BS4DataProvider',
    
    # 东方财富提供者
    'EastMoneyFundProvider',
    'EastMoneyStockProvider',

    # 基金估值提供者
    'FundEstimateProvider',
]

# 模块元信息
__module_name__ = "data_provider"
__module_description__ = "统一的金融数据提供者接口"
__supported_sources__ = [
    "东方财富 (EastMoney)",
    "自定义爬虫",
    "BS4网页解析"
]

# 数据提供者注册表
AVAILABLE_PROVIDERS = {
    'eastmoney_fund': EastMoneyFundProvider,
    'eastmoney_stock': EastMoneyStockProvider,
    'fund_estimate': FundEstimateProvider,
}

def get_provider(provider_name: str, *args, **kwargs):
    """获取指定名称的数据提供者实例。

    Parameters
    ----------
    provider_name
        注册表中的数据提供者键名。
    *args, **kwargs
        直接传递给数据提供者类的构造函数，用于控制初始化逻辑。

    Returns
    -------
    DataProvider
        数据提供者实例。

    Raises
    ------
    ValueError
        当提供者名称未注册时抛出。
    """
    try:
        provider_cls = AVAILABLE_PROVIDERS[provider_name]
    except KeyError as exc:
        raise ValueError(f"未知的数据提供者: {provider_name}") from exc

    return provider_cls(*args, **kwargs)

# 添加到导出列表
__all__.append('get_provider')
__all__.append('AVAILABLE_PROVIDERS')