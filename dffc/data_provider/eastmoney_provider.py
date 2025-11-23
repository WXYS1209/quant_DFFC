"""
东方财富基金数据提供者

实现从东方财富获取资产数据的功能
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import pandas as pd

from .base import BS4DataProvider, DataProviderConfig
from .._utils import DataFetchError, validate_fund_code, validate_stock_code, safe_float_convert


class EastMoneyFundProvider(BS4DataProvider):
    """
    东方财富基金数据提供者
    
    从东方财富基金接口获取基金净值等相关数据
    """
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        if config is None:
            config = DataProviderConfig(
                timeout=30,
                retry_count=3,
                page_size=49,  # 东方财富默认每页49条记录
                rate_limit=0.5,  # 0.5秒间隔，避免请求过于频繁
                base_url="http://fund.eastmoney.com/f10/F10DataApi.aspx"
            )
        super().__init__(config)
    
    def fetch_raw_data(self, code: str, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        从东方财富接口获取原始基金数据
        
        Args:
            code: 基金代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            原始数据列表
            
        Raises:
            DataFetchError: 数据获取失败
            ValidationError: 参数验证失败
        """
        # 验证基金代码
        validated_code = validate_fund_code(code)
        
        date_fmt = "%Y-%m-%d"

        if start_date is None or end_date is None:
            # 如果没有提供日期范围，获取最近数据（默认30天）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

        base_params = {
            "type": "lsjz",
            "code": code,
            "per": self.config.page_size,
            "sdate": start_date.strftime(date_fmt),
            "edate": end_date.strftime(date_fmt),
        }
        
        all_data = []
        page = 0
        max_pages = 100  # 防止无限循环
        
        while page < max_pages:
            page += 1
            params = base_params.copy()
            params["page"] = page
            
            try:
                # 请求间隔控制
                if page > 1:
                    time.sleep(self.config.rate_limit)

                response = self._make_request(self.config.base_url, params)
                page_data = self._parse_html_response(response.text)
                
                if not page_data:  # 没有更多数据
                    break
                    
                all_data.extend(page_data)
                
            except Exception as e:
                raise DataFetchError(f"Failed to fetch data for fund {code}, page {page}: {str(e)}")
        
        return all_data
    
    @staticmethod
    def _parse_html_response(html_content: str) -> List[Dict[str, Any]]:
        """
        解析HTML响应内容
        
        Args:
            html_content: HTML内容
            
        Returns:
            解析后的数据列表
        """
        soup = BeautifulSoup(html_content, 'lxml')
        data_list = []
        th_list = None
        
        for idx, tr in enumerate(soup.find_all('tr')):
            if idx == 0:
                # 第一行是表头
                th_list = [x.text for x in tr.find_all("th")]
            else:
                # 数据行
                tds = tr.find_all('td')
                if not tds:
                    continue
                    
                values = [w.text for w in tds]
                if values and values[0] == "暂无数据!":
                    break
                
                if th_list and len(values) == len(th_list):
                    dict_data = dict(zip(th_list, values))
                    data_list.append(dict_data)
        
        return data_list
    
    def parse_data(self, raw_data):
        df = pd.DataFrame(raw_data)
        df.rename(
            columns={
                '净值日期': 'date',
                '单位净值': 'unit_value',
                '累计净值': 'cumulative_value',
                '日增长率': 'daily_growth_rate', 
                '申购状态': 'purchase_state', 
                '赎回状态': 'redemption_state',
                '分红送配': 'bonus_distribution'
            }, 
            inplace=True
        )

        df[['unit_value', 'cumulative_value', 'daily_growth_rate']] = df[['unit_value', 'cumulative_value', 'daily_growth_rate']].map(safe_float_convert)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
        
        return df
    
    @property
    def name(self) -> str:
        """提供者名称"""
        return "EastMoney Fund Provider"
    
    @property
    def description(self) -> str:
        """提供者描述"""
        return "Fetches fund data from East Money fund API"


class EastMoneyStockProvider(BS4DataProvider):
    """
    东方财富股票数据提供者
    
    从东方财富股票接口获取股票净值等相关数据
    """
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        if config is None:
            config = DataProviderConfig(
                timeout=30,
                retry_count=3
            )
        super().__init__(config)
    
    def fetch_raw_data(self, code: str, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        从东方财富接口获取原始股票数据
        
        Args:
            code: 股票代码
            start_date: 开始日期 (可选，如果不提供则获取当天数据)
            end_date: 结束日期 (可选，如果不提供则获取当天数据)
            
        Returns:
            原始数据列表
            
        Raises:
            DataFetchError: 数据获取失败
            ValidationError: 参数验证失败
        """
        # 验证股票代码
        validated_code = validate_stock_code(code)
        
        if start_date is None or end_date is None:
            # 如果没有提供日期范围，获取最近数据（默认30天）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

        # 确定市场代码：沪市=1，深市=0
        if code.startswith('6') or code.startswith('51'):  # 沪市股票和ETF
            market = "1"
        elif code.startswith(('0', '3', '15')):  # 深市股票
            market = "0"
        else:
            market = "1"  # 默认沪市
        
        # 计算需要获取的数据量（交易日大约每年250天）
        total_days = (end_date - start_date).days
        klt = 101  # 日K线
        lmt = min(max(total_days + 50, 100), 1000)  # 限制在100-1000之间
        
        # 东方财富K线数据接口
        self.config.base_url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get"
        
        params = {
            'secid': f"{market}.{code}",
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': klt,  # K线类型：101=日K，102=周K，103=月K
            'fqt': 1,    # 复权类型：0=不复权，1=前复权，2=后复权
            'lmt': lmt,  # 数据量
            'end': '20500101',  # 结束日期，设置为未来日期获取最新数据
            'iscca': 1   # 是否需要日线前收盘价
        }
        
        try:
            response = self._make_request(self.config.base_url, params)
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return []
            
            klines = data['data'].get('klines', [])
            if not klines:
                return []
            
            result = []
            for kline in klines:
                # 数据格式：日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
                parts = kline.split(',')
                if len(parts) < 11:
                    continue
                
                try:
                    trade_date = datetime.strptime(parts[0], '%Y-%m-%d')
                    
                    # 过滤日期范围
                    if trade_date < start_date or trade_date > end_date:
                        continue
                    
                    open_price = float(parts[1])
                    close_price = float(parts[2])
                    high_price = float(parts[3])
                    low_price = float(parts[4])
                    volume = int(parts[5]) if parts[5] else 0
                    amount = float(parts[6]) if parts[6] else 0.0
                    change_percent = float(parts[8]) if parts[8] else 0.0
                    change = float(parts[9]) if parts[9] else 0.0
                    
                    result.append({
                        'code': code,
                        'name': data['data'].get('name', ''),
                        'date': trade_date.strftime('%Y-%m-%d'),
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'amount': amount,
                        'change': change,
                        'change_percent': change_percent,
                        'timestamp': trade_date,
                        'source': 'eastmoney_kline'
                    })
                    
                except (ValueError, IndexError) as e:
                    # 跳过无效数据
                    continue
            
            # 按日期排序（最新的在前）
            result.sort(key=lambda x: x['timestamp'], reverse=True)
            return result
            
        except Exception as e:
            raise DataFetchError(f"Failed to fetch historical data for stock {code}: {str(e)}")
    
    def parse_data(self, raw_data):
        df = pd.DataFrame(raw_data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
        return df
    
    @property
    def name(self) -> str:
        """提供者名称"""
        return "EastMoney Stock Provider"
    
    @property
    def description(self) -> str:
        """提供者描述"""
        return "Fetches stock data from East Money stock API"
