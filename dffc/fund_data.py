"""
基金数据提供者 - vectorbt Data 子类实现

将现有的基金数据提供者功能集成到vectorbt框架中
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Sequence, Hashable
import vectorbt as vbt
from vectorbt.data.base import Data
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.datetime_ import to_tzaware_datetime

from vectorbt import _typing as tp

from dffc.data_provider.base import DataProvider
from dffc.data_provider.eastmoney_provider import EastMoneyFundProvider
from dffc._utils import validate_fund_code

class FundData(Data):
    """
    基金数据类 - vectorbt Data 的子类
    
    提供基金净值、增长率等数据的获取和管理功能
    类似于 vbt.BinanceData 的实现方式
    
    默认时区设置为北京时间 (Asia/Shanghai)
    """
    
    _expected_keys = ('unit_value', 'cumulative_value', 'daily_growth_rate',
                     'purchase_state', 'redemption_state', 'bonus_distribution')
    
    # 默认时区设置
    _default_timezone = 'Asia/Shanghai'  # 北京时间
    
    @classmethod
    def download_symbol(cls, 
                       symbol: Hashable,
                       provider: Optional[DataProvider] = None,
                       start: tp.DatetimeLike = 0,
                       end: tp.DatetimeLike = 'now',
                       **kwargs) -> pd.DataFrame:
        """
        下载单个基金的数据
        
        Args:
            symbol: 基金代码
            provider: 数据提供者实例
            **kwargs: 其他参数（start_date, end_date等）
            
        Returns:
            包含基金数据的DataFrame，索引为北京时间
        """
        if provider is None:
            raise ValueError("provider must be provided")
        
        # 验证基金代码
        validated_symbol = validate_fund_code(symbol)
        
        # 处理日期参数
        start_date = to_tzaware_datetime(start)
        end_date = to_tzaware_datetime(end)

        # 获取数据
        df = provider.get_data(validated_symbol, start_date, end_date)
        
        # 确保包含必要的列
        for key in cls._expected_keys:
            if key not in df.columns:
                df[key] = np.nan
        
        # 确保索引是 DatetimeIndex，如果不是则转换
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
            
    @classmethod 
    def download(cls,
                symbols: Union[Hashable, Sequence[Hashable]],
                provider: Optional[EastMoneyFundProvider] = None,
                **kwargs) -> "FundData":
        """
        下载多个基金的数据，类似于 BinanceData.download 的实现方式
        
        Args:
            symbols: 基金代码或基金代码列表
            provider: 数据提供者实例，如果为None则自动创建
            **kwargs: 其他参数，包括：
                - start_date: 开始日期
                - end_date: 结束日期
                - missing_index: 缺失索引的处理方式 ('ignore', 'raise', 'drop')
                - missing_columns: 缺失列的处理方式 ('ignore', 'raise', 'drop')
                - tz_localize: 时区本地化，默认使用类的默认时区（北京时间）
                - tz_convert: 时区转换
                - 以及 provider 的初始化参数
            
        Returns:
            FundData实例
        """
        from vectorbt.utils.config import get_func_kwargs
        
        # 提取 provider 相关的参数
        provider_kwargs = dict()
        try:
            for k in get_func_kwargs(EastMoneyFundProvider.__init__):
                if k in kwargs and k != 'self':
                    provider_kwargs[k] = kwargs.pop(k)
        except:
            # 如果 get_func_kwargs 不可用，手动处理常见参数
            common_provider_params = ['headers', 'timeout', 'retry_times']
            for param in common_provider_params:
                if param in kwargs:
                    provider_kwargs[param] = kwargs.pop(param)
        # 处理基金名称
        if "names" in kwargs:
            names = kwargs.pop("names")
            fund_names = {}
            # 确保symbols是列表格式
            if isinstance(symbols, (str, int)):
                symbols_list = [symbols]
            else:
                symbols_list = list(symbols)
            
            if isinstance(names, str):
                # 单个名称，对应单个基金
                if len(symbols_list) == 1:
                    fund_names[str(symbols_list[0])] = names
                else:
                    raise ValueError("Single name provided but multiple symbols given")
            elif isinstance(names, dict):
                # 字典映射
                fund_names = {str(k): v for k, v in names.items()}
            elif isinstance(names, list):
                # 名称列表，与symbols顺序对应
                if len(names) != len(symbols_list):
                    raise ValueError("Length of names list must match length of symbols")
                fund_names = {str(symbol): name for symbol, name in zip(symbols_list, names)}
            cls.names = fund_names

        # 如果没有提供 provider，则自动创建
        if provider is None:
            provider = EastMoneyFundProvider(**provider_kwargs)
        
        # 提取 Data.download 的参数
        data_download_kwargs = {}
        data_download_params = ['missing_index', 'missing_columns', 'tz_localize', 'tz_convert', 'wrapper_kwargs']
        for param in data_download_params:
            if param in kwargs:
                data_download_kwargs[param] = kwargs.pop(param)
        
        # 如果没有指定时区，使用类的默认时区
        if 'tz_localize' not in data_download_kwargs:
            data_download_kwargs['tz_localize'] = cls._default_timezone
            data_download_kwargs['tz_convert'] = cls._default_timezone
        
        # 调用父类的 download 方法
        return super(FundData, cls).download(symbols, provider=provider, **data_download_kwargs, **kwargs)
    
    def get(self, column: Optional[str] = None, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        重写get方法，确保返回的DataFrame保持列名
        
        Args:
            column: 指定列名，如果为None则返回包含列名的字典
            
        Returns:
            DataFrame或包含列名的字典
        """
        # 如果只有一个symbol，直接返回
        if len(self.symbols) == 1:
            if column is None:
                return self.data[self.symbols[0]]
            return self.data[self.symbols[0]][column]
        
        # 多个symbols时，使用concat方法
        concat_data = self.concat(**kwargs)
        
        if column is not None:
            if isinstance(column, list):
                return {c: concat_data[c] for c in column}
            return concat_data[column]
        
        # 返回包含列名的字典，而不是元组
        return concat_data
    
    def get_fund_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取基金基本信息
        
        Args:
            symbol: 基金代码
            
        Returns:
            基金信息字典
        """
        if symbol not in self.symbols:
            raise ValueError(f"Fund {symbol} not found in data")
        
        fund_data = self.get('unit_value')[symbol]
        
        if self.names:
            fund_name = self.names.get(str(symbol), "")
        return {
            'symbol': symbol,
            'name': fund_name if self.names else "",
            'start_date': fund_data.index.min(),
            'end_date': fund_data.index.max(),
            'total_days': len(fund_data),
            'latest_unit_value': fund_data.iloc[-1] if len(fund_data) > 0 else None
        }
    
    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.Frame:
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `YFData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        download_kwargs['end'] = "now"
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)
    
    @property
    def provider_name(self) -> str:
        """数据提供者名称"""
        return "EastMoney Fund Provider"
    
    def __repr__(self) -> str:
        return f"<FundData: {len(self.symbols)} funds, {self.provider_name}>"

