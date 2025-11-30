# dffc 技术参考

`dffc` 基于 vectorbt 提供了一套聚焦中国公募基金研究的工具链。它统一了数据采集、Holt-Winters 建模、策略编排与回测接口，使研究型 Notebook 与生产级应用可以共用同一套基础组件。本文档逐一梳理包内元素、解决的问题以及对外暴露的接口，方便团队协作和后期扩展。

## 包结构速览

| 路径 | 职责 | 关键产物 |
| --- | --- | --- |
| `dffc/__init__.py` | 包入口与稳定 API 聚合 | `FundData`、`HW`、`ReallocationBackTest`、公共异常 |
| `dffc/_utils.py` | 输入校验与通用工具 | `ValidationError`、`parse_date`、`safe_float_convert` |
| `dffc/fund_data.py` | 向量化基金数据容器 | `FundData.download`、`FundData.get`、元数据接口 |
| `dffc/data_provider/` | 数据源抽象 + 东方财富实现 | `DataProvider`、`EastMoneyFundProvider`、`EastMoneyStockProvider` |
| `dffc/estimate_utils.py` | 估值补全工具 | `append_estimates_to_prices` |
| `dffc/holt_winters/` | Holt-Winters 指标与优化器 | `HW`、`HWDP`、`HWOptimizer`、绘图助手 |
| `dffc/strategies/` | 策略基类与再平衡框架 | `Strategy`、`ReallocationStrategy` |
| `dffc/backtesting/` | 轻量回测编排器 | `BackTest`、`ReallocationBackTest` |

各子目录仅导出必要的类型化 API，可直接 `from dffc import HW` 引用，不泄露实现细节。

> 📚 相关官方文档：
> - vectorbt: https://vectorbt.dev/
> - Streamlit: https://docs.streamlit.io/

## 模块详解

### 核心工具 (`_utils.py`)
- **职责**：统一管理基金/股票代码、日期、数值等输入的校验与解析，供数据提供者与策略复用。
- **主要接口**：
  - `parse_date(date_str, fmt="%Y-%m-%d") -> datetime`：严格的日期解析，出错时返回详细提示。
  - `validate_date_range(start, end)`：确保开始时间早于结束时间，用于请求前置校验。
  - `validate_fund_code(code)` / `validate_stock_code(code)`：统一 6 位基金代码与带 SH/SZ 后缀的股票代码。
  - `safe_float_convert(value, default=None)`：处理百分号、中文标点、缺失符号等，安全转换为 `float`。
- **异常体系**：`ValidationError`（输入异常）、`DataFetchError`（网络/抓取异常），上层模块统一抛出，易于集中处理。

### 基金数据容器 (`fund_data.py`)
- **职责**：将多只基金的时序数据封装在 vectorbt `Data` 子类中，提供时区感知索引与友好的访问接口。
- **关键方法**：
  - `FundData.download(symbols, provider=None, **kwargs)`：批量下载基金数据，若未传 `provider`，自动实例化 `EastMoneyFundProvider`，支持 `names` 参数绑定基金名称。
  - `FundData.download_symbol(symbol, provider, start, end, **kwargs)`：与 vectorbt `Data` 基类保持一致的下载钩子。
  - `FundData.get(column=None)`：返回单列/多列数据时保留原始列名，避免 tuple 风格列头。
  - `FundData.get_fund_info(symbol)`：聚合历史区间、可用天数、最新净值等摘要信息。
  - `FundData.update_symbol(symbol, **kwargs)`：增量更新单只基金，起始日期默认取上一条数据末端。
- **实现要点**：默认将数据本地化至 `Asia/Shanghai`，`FundData.names` 可维护代码与名称映射，方便报表展示。

### 数据提供者 (`data_provider`)
- **职责**：隔离 HTTP 请求与解析逻辑，向上层提供结构化 `DataFrame`，不耦合业务语义。
- **组件**：
  - `DataProviderConfig`：记录 `timeout`、`retry_count`、`page_size`、`rate_limit`、`headers` 等网络参数。
  - `DataProvider` 抽象基类：定义 `fetch_raw_data`、`parse_data`、`get_data`，并内建日期区间校验。
  - `BS4DataProvider`：提供带重试/退避的 GET 辅助方法，适配 HTML 响应解析（BeautifulSoup）。
- **具体实现**：
  - `EastMoneyFundProvider`：分页抓取东方财富基金历史净值，将原始表头转换为 `unit_value`、`daily_growth_rate` 等标准列，并通过 `safe_float_convert` 清洗数值，最终按时间顺序输出。
  - `EastMoneyStockProvider`：对接东方财富 K 线 JSON 接口，自动判断市场前缀（沪 `1.`、深 `0.`），返回 OHLCV 数据，兼容跨资产研究。

### 估值补全 (`estimate_utils.py`)
- **职责**：将估值数据与日频净值对齐，补齐缺失日期。
- **核心函数**：`append_estimates_to_prices(price_data, codes, provider, value_column="estimate_value", timestamp_column="estimate_timestamp", target_timezone=None)`
  - 校验 `price_data` 必须使用 `DatetimeIndex`。
  - 逐只基金通过 `provider.get_data` 获取估值，检查字段齐全性，按目标时区规范化时间戳。
  - 仅附加未出现的日期，返回排序后的副本；若估值缺失则抛出异常，便于上游监控处理。

### Holt-Winters 系列 (`holt_winters`)
- **指标层 (`_holt_winters.py`)**：
  - `HW`、`HWD`、`HWDP`：基于 numba 内核（`holt_winters_ets_1d_nb`、`hw_delta_nb`、`hw_delta_percentage_nb`）构建的 `IndicatorFactory`。调用 `.run(close, alpha, beta, gamma, season_length, multiplicative=False)` 即可得到命名输出（`hw`、`hwd`、`hwdp`）。乘法模型会验证正值输入并即时检查平滑参数范围。
  - 递推实现遵循 ETS(A, A, A/M) 形式，针对金融序列改进了初始 level/trend/seasonal 的设定。
- **优化层 (`_optimization.py`)**：
  - `HWOptimizer(original_data: DataFrame)`：面向多资产的批量参数搜索，可选 `end_day` 窗口用于观察稳定性。
    - `optimize(...)`：按季节长度并行搜索（`ProcessPoolExecutor`），支持输出单资产 JSON 与汇总文件。
    - `update_params`、`get_asset`、`get_summary`、`save_summary`、`from_summary`：满足仪表盘或 CLI 管理参数的需求。
  - `HWAssetResult`：封装单资产最优解序列，提供 `analyze_stability`、`plot_hw_opt`、`compute_hw_curves` 以及序列化方法，Notebook/Streamlit 可直接复用。

### 策略框架 (`strategies`)
- **策略基类**：`Strategy(prices)` 仅持有原始价格与 `backtest_prices`，定义 `run_backtest` 抽象接口。
- **再平衡基类**：`ReallocationStrategy(Strategy)` 提供可复用的目标权重执行逻辑：
  - 参数解析工具 `_parse_numeric_param`、`_parse_array_param` 支持标量、数组、网格输入，方便做参数遍历。
  - 执行辅助方法：
    - `_apply_gradual_adjustment`：带容差的渐进调仓；
    - `_apply_trade_delay`：按 T+N 模拟下单延迟；
    - `_weights_to_orders` / `_prepare_orders`：生成 vectorbt `TargetPercent` 订单矩阵。
  - `_get_param_combinations`、`_generate_target_weights` 需在子类中实现，如 `wxy_backtest/strategy.py` 内的 `DualReallocationStrategy`。
  - `run_backtest(...)`：为每组参数复制价格矩阵，可选并行计算目标权重，最终返回 `SimpleNamespace`，内含 portfolio、实际权重、调仓标记与参数元数据，便于后续统计与绘图。

### 回测编排 (`backtesting`)
- **`BackTest`**：任何策略的轻量包装器。负责合并默认参数、调用 `strategy.run_backtest`，缓存 portfolio (`self.pf`)，并通过 `_multi` 标识是否存在多参数组。
  - `run(**kwargs)`：执行策略。
  - `stats(selected=None)`：单组返回 `pf.stats()`；多组则合并参数元数据与核心指标（`total_return`、`sharpe_ratio`、`max_drawdown` 等）为 MultiIndex DataFrame。
  - `plot(**kwargs)`：抽象方法，由子类具体实现。
- **`ReallocationBackTest`**：面向再平衡策略的可视化与筛选工具。
  - `plot(index_levels=None, column_levels=None)`：单组时输出价格/买卖点/权重/收益的多联图；多组时绘制热力图仪表盘，需显式指定或自动推断 MultiIndex 维度。
  - `get_best_param`、`get_weighted_best_params`：基于指标（或加权指标）筛选最优参数组合，支持研究自动化流程。

## 工作流示例

1. **下载基金数据**
   ```python
   from dffc import FundData, EastMoneyFundProvider

   provider = EastMoneyFundProvider()
   fund_data = FundData.download(['007467', '004253'], provider=provider)
   price_panel = fund_data.get('unit_value')
   ```

2. **（可选）拼接最新估值**
   ```python
   from dffc import append_estimates_to_prices, FundEstimateProvider

   estimate_provider = FundEstimateProvider()
   price_panel = append_estimates_to_prices(price_panel, ['007467', '004253'], estimate_provider)
   ```

3. **运行 Holt-Winters 模型或优化器**
   ```python
   from dffc import HW, HWOptimizer

   hw_single = HW.run(price_panel['004253'], 0.12, 0.02, 0.08, 12, multiplicative=False)

   optimizer = HWOptimizer(price_panel[['004253', '007467']])
   hw_summary = optimizer.optimize(save=False)
   ```

4. **执行策略回测**
   ```python
   from my_project.strategy import DualReallocationStrategy
   from dffc import ReallocationBackTest

   strategy = DualReallocationStrategy(prices=price_panel, hw_params_list=hw_summary)
   backtest = ReallocationBackTest(strategy=strategy, start_date='2022-01-01', initial_cash=100_000, trade_delay=1)
   backtest.run()
   print(backtest.stats())
   fig = backtest.plot(column_levels=['threshold'], index_levels=['adjust_factor'])
   fig.show()
   ```

本指南旨在帮助贡献者与使用者迅速定位目标模块、理解其接口并组合端到端的研究流程。若未来新增模块或公共入口，请同步扩展本文档以保持一致性。
