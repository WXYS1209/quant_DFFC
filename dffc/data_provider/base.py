"""
数据提供者抽象接口

定义数据获取、缓存和存储的抽象接口
"""

import requests
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd
from dffc import validate_date_range, DataFetchError

@dataclass
class DataProviderConfig:
    """数据提供者配置"""
    timeout: int = 30
    retry_count: int = 3
    page_size: int = 50
    headers: Optional[dict] = None
    base_url: Optional[str] = None
    rate_limit: float = 0.1  # 请求间隔（秒）

class DataProvider(ABC):
    """数据提供者抽象基类"""
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        self.config = config or DataProviderConfig()
    
    @abstractmethod
    def fetch_raw_data(self, code: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """获取原始数据"""
        pass
    
    @abstractmethod
    def parse_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        将原始数据解析为pandas DataFrame对象
        
        Args:
            raw_data: 原始数据列表
            
        Returns:
            pandas DataFrame对象
        """
        pass
    
    def get_data(self, code: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """获取资产数据的主入口方法"""
        if start_date and end_date:
            start_date, end_date = validate_date_range(start_date, end_date)
        raw_data = self.fetch_raw_data(code, start_date, end_date)
        return self.parse_data(raw_data)

    @property
    def provider_name(self) -> str:
        """数据提供者名称"""
        return self.__class__.__name__


class BS4DataProvider(DataProvider):
    """
    基于HTTP的数据提供者基类
    
    为需要HTTP请求的数据提供者提供通用功能
    """
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        super().__init__(config)
        
    def _make_request(self, url: str, params: Dict[str, Any]=None) -> requests.Response:
        """
        发起HTTP请求
        
        Args:
            params: 请求参数
            
        Returns:
            响应对象
            
        Raises:
            DataFetchError: 请求失败
        """
        for attempt in range(self.config.retry_count):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.config.timeout,
                    headers=self.config.headers
                )
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                if attempt == self.config.retry_count - 1:
                    raise DataFetchError(f"Request failed after {self.config.retry_count} attempts: {str(e)}")
                time.sleep(1)  # 重试前等待
