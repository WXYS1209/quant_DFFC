from typing import Union, Any, Optional
from datetime import datetime
import re

def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime:
    """
    解析日期字符串为datetime对象
    
    Args:
        date_str: 日期字符串
        fmt: 日期格式
    
    Returns:
        datetime对象
    
    Raises:
        ValidationError: 解析失败时抛出
    """
    try:
        return datetime.strptime(date_str, fmt)
    except ValueError as e:
        raise ValidationError(f"Failed to parse date '{date_str}' with format '{fmt}': {e}")


def validate_date_range(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> tuple:
    """
    验证日期范围的有效性
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        (start_datetime, end_datetime) 元组
    
    Raises:
        ValidationError: 日期范围无效时抛出
    """
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)
    
    if start_date >= end_date:
        raise ValidationError(f"Start date {start_date} must be before end date {end_date}")
    
    return start_date, end_date


def validate_fund_code(code: Any) -> str:
    """
    验证基金代码
    
    Args:
        code: 基金代码输入
    
    Returns:
        验证后的基金代码字符串
    
    Raises:
        ValidationError: 基金代码无效时抛出
    """
    if not code:
        raise ValidationError("Fund code cannot be empty")
    
    if not isinstance(code, str):
        code = str(code)
    
    # 移除空格
    code = code.strip()
    
    # 基金代码应该是6位数字
    if not code.isdigit():
        raise ValidationError(f"Fund code must contain only digits: {code}")
    
    if len(code) != 6:
        raise ValidationError(f"Fund code must be 6 digits: {code}")
    
    return code

def validate_stock_code(code: Any) -> str:
    """
    验证股票代码
    
    Args:
        code: 股票代码输入
    
    Returns:
        验证后的股票代码字符串
    
    Raises:
        ValidationError: 股票代码无效时抛出
    """
    if not code:
        raise ValidationError("Stock code cannot be empty")
    
    if not isinstance(code, str):
        code = str(code)
    
    code = code.strip().upper()
    
    # 支持A股代码格式: 6位数字 或 6位数字.SH/SZ
    if re.match(r'^\d{6}$', code):
        return code
    elif re.match(r'^\d{6}\.(SH|SZ)$', code):
        return code
    else:
        raise ValidationError(f"Invalid stock code format: {code}")

def safe_float_convert(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    安全转换为float类型
    
    Args:
        value: 要转换的值
        default: 转换失败时的默认值
    
    Returns:
        转换后的float值或默认值
    """
    if value is None or value == '' or value == '--' or value == 'N/A':
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # 移除百分号和空格
        value = value.strip()
        
        # 处理中文标点
        value = value.replace('，', '').replace('－', '-')
        
        try:
            if "%" in value:
                return float(value.replace("%", "")) / 100
            return float(value)
        except (ValueError, TypeError):
            return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class ValidationError(Exception):
    """数据验证异常"""
    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value

class DataFetchError(Exception):
    """数据获取异常"""
    def __init__(self, message: str, url: str = None, params: dict = None):
        super().__init__(message)
        self.url = url
        self.params = params
