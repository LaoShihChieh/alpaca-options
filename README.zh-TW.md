# alpaca-options 🦙

一個**僅限模擬交易**的 Python 選擇權策略工具包，基於 [Alpaca Markets](https://alpaca.markets/) 平台，以 [`alpaca-py`](https://github.com/alpacahq/alpaca-py) 為底層建構。

---

## 環境需求

| 工具 | 版本 |
|------|------|
| Python | ≥ 3.12 |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.4 |
| Alpaca 模擬帳戶 | — |

---

## 安裝步驟

```bash
# 1. 進入專案目錄
cd alpaca-options

# 2. 安裝 uv（若尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 確認憑證已寫入 .env
cat .env
# ALPACA_API_KEY=...
# ALPACA_SECRET_KEY=...
# ALPACA_PAPER=true

# 4. 安裝相依套件
uv sync
```

> **模擬交易保護機制**：程式庫在初始化客戶端時會強制確認 `ALPACA_PAPER=true`。
> 程式碼絕不會在真實帳戶上執行。

---

## 執行範例

### 買入價平買權（單腳）

```bash
uv run python examples/buy_atm_call.py
```

### 牛市買權價差

```bash
uv run python examples/vertical_spread.py
```

### 風險管理示範（無 API 呼叫，純邏輯驗證）

```bash
uv run python examples/risk_manager_demo.py
```

### 鐵禿鷹 90 天回測

```bash
uv run python examples/backtest_90_days.py
# 執行結果會將資產曲線圖片 equity_curve.png 存於當前目錄
```

### 即時空跑（會要求確認，不提交任何委託單）

```bash
uv run python examples/live_dry_run.py
```

### 執行測試

```bash
uv run pytest -v
```

---

## 鐵禿鷹 0DTE 策略

### 設計概述

本策略針對 SPY 標的，在現貨價格下方賣出賣權價差、上方賣出買權價差，兩者均於**當日**到期（0DTE）。由於 SPY 選擇權每個交易日均有到期合約，永遠存在可操作的 0DTE 鏈。

```
            put_long  put_short   現貨   call_short  call_long
權利金：      $0.40     $1.00    580      $1.00       $0.40
              ←←← 長翼 ←←←              →→→ 長翼 →→→
淨收取權利金 = ($1.00 + $1.00) − ($0.40 + $0.40) = 每股 $1.20
```

#### 進場條件（所有條件須同時滿足）

| # | 規則 | 數值 |
|---|------|------|
| 1 | 時間窗口 | 美東時間 10:00–14:00 |
| 2 | VIX | < 25 |
| 3 | 非重大事件日 | FOMC、CPI、NFP 日跳過 |
| 4 | 最低權利金 | ≥ 翼距的 8% |
| 5 | 短腳 Delta 值 | ≈ 0.10（Black-Scholes 計算） |
| 6 | 翼距 | $5 |

#### 出場條件（先觸發者優先）

| 觸發條件 | 說明 |
|----------|------|
| 獲利目標 | 價差成本 ≤ 開倉時權利金的 50% |
| 停損 | 價差成本 ≥ 開倉時權利金的 2 倍 |
| Delta 突破 | 短腳 Delta ≥ 0.25 |
| 時間 | 美東時間 15:30 強制平倉 |

---

## 為何先進行回測——期望值數學

未加過濾的 0DTE 鐵禿鷹勝率約為 **75%**（SPY 收盤價維持在短腳履約價之間）。但虧損的 25% 會產生相當於已收權利金 2–5 倍的損失：

```
未過濾期望值 ≈ 0.75 × $100 權利金  −  0.25 × $300 平均虧損  ≈  $0
```

在扣除手續費與滑點之前便接近損益兩平，實際上是負期望值。過濾機制創造了優勢：

1. **VIX 過濾** — 跳過波動率最高的交易日，這類日子選擇權定價昂貴，且標的波動幅度最大（鐵禿鷹最容易爆倉的環境）。
2. **事件過濾** — FOMC、CPI、NFP 日呈現肥尾分布，Black-Scholes 模型無法反映。跳過這些日子可排除最極端的尾部風險。
3. **最低權利金過濾** — 只在市場為既有風險定出足夠溢價時才進場。低權利金的鐵禿鷹報酬微薄，損失卻是全額。
4. **熔斷機制** — 日損失上限與 10% 最大回撤止損，防止單週虧損演變為災難性的單月損失。

回測的目的，在於測量哪些過濾參數組合在測試期間真正產生了正風險調整報酬。

> ⚠️ **回測樂觀偏差**：成交價以 K 線中間價計算。
> 由於買賣價差滑點、限價單排隊優先順序，以及 IV 模型誤差（固定 IV 的 Black-Scholes ≠ 真實波動率微笑），預期實際交易結果會**差 10–20%**。

---

## 優勢從何而來

本策略處於 Gamma 和 Vega 空頭部位。這種部位在散戶選擇權交易中的基準勝率為負值：大多數賣出期權的交易者會在一年內爆倉。因此問題不是「賣出權利金是否有效」（平均而言並不有效），而是「我具體做了什麼，使期望值轉為正值？」

按重要性排序，有三件事：

### 1. 事件日曆過濾（最大優勢來源）

在非事件日，SPY 0DTE 的日內走勢大致呈常態分布，標準差偏小，典型日內波動幅度約 0.5–0.8%。本策略的 10 Delta 短腳履約價遠在此分布之外，因此 75–85% 的理論勝率是切實可行的。

在事件日（FOMC、CPI、NFP），分布出現肥尾。一個在正常日屬於 4 個標準差的 2% 走勢，在 FOMC 日可能只是 1 個標準差的事件。在這種分布下賣出 10 Delta 勒式，是截然不同的賽局，期望值也截然不同——而且更差。

過濾約 13% 的交易日（事件日）並不會讓報酬減少 13%。它所削減的尾部風險遠不止如此，因為事件日承載著不成比例的尾部權重。這是本策略得以獲利的最主要原因。

### 2. 最低權利金門檻（第二大優勢來源）

在低波動日，一個 10 Delta 鐵禿鷹可能以 $5 翼距只收到 $0.15 的權利金。這是 32:1 的風險報酬比，需要約 97% 的勝率才能打平。不可行。

8% 的「權利金佔翼距比例」最低門檻會自動過濾掉這類日子。它能自我調整至波動率環境：當 VIX 偏低時，市場給出的溢價稀薄，更多日子被過濾掉；當 VIX 偏高（但低於 25 的環境切換閾值）時，溢價較豐厚，更多日子符合資格。這個過濾機制在不引入顯式環境邏輯的情況下，將靜態策略轉化為具環境感知能力的策略。

### 3. 鐵禿鷹結構的有限風險（生存防護層）

前兩個過濾機制創造了正期望值。鐵禿鷹結構確保這個期望值有機會在足夠多的交易中收斂實現。

裸賣勒式的兩側風險均為無限。任何一個 SPY 走勢達 4% 的尾部日——雖然罕見，但確實存在：2024 年 8 月日圓套利交易崩解、2020 年 3 月新冠疫情暴跌、2018 年 2 月 Volmageddon 事件——都可能抹去數月累積的權利金收入。即使期望值為正，策略也可能在獲利實現前便遭到淘汰。

鐵禿鷹將最大損失上限定為翼距減去權利金。以收取 $0.40 權利金的 $5 翼距為例，每口合約最差的損失為 $4.60。有界。策略即便在尾部日判斷失誤，隔天早上依然能繼續運作。

這就是為何鐵禿鷹的夏普比率通常高於裸賣勒式，儘管其單筆期望報酬較低——生存能力本身會帶來複利效應。

### 什麼不是優勢來源

- **在 10 Delta 範圍內調整履約價。** 從 0.08 Delta 調整至 0.12 Delta 能讓勝率改變幾個百分點，但不會從根本上改變期望值的數學。
- **調整出場時機**（50% 獲利目標 vs. 40% vs. 60%）。這些只是在特定環境內的優化，無法讓虧損策略轉為獲利。
- **在溫和市場期間回測。** 低波動率單向趨勢市場中呈現 100% 勝率的回測，對於實際交易表現毫無參考價值。

### 優勢可能消失的情境

- **事件日曆資料來源失效**（FRED 服務中斷、硬編碼日期過期）：事件日滑過過濾機制，導致策略在已知波動衝擊下建倉。
- **VIX 環境閾值漂移。** 25 在歷史上是合理的，但未來未必適用。
- **市場微結構變化。** 若選擇權造市商縮窄買賣價差，中間價與實際成交價之間的空間就會縮小，侵蝕權利金假設。
- **行為覆蓋。** 交易者在某個「感覺安全」的日子繞過過濾機制，從根本上違背了策略設計的初衷。

**季度檢視清單：**

- 確認 FRED 日曆與勞工統計局（BLS）及聯準會（Fed）公布的時間表一致
- 重新核對 VIX 環境閾值，對照近期已實現波動率
- 比較回測成交與實際成交；若滑點超過 20%，重新校準

---

## 風險控制

> 以下控制機制用於防範策略的失效情境（失效情境的完整說明請參閱上方〈[優勢從何而來](#優勢從何而來)〉）。

| 控制項目 | 預設值 | 作用 |
|----------|--------|------|
| `vix_threshold` | 25 | VIX ≥ 25 時不進場 |
| `max_drawdown_pct` | 10% | 熔斷機制：投資組合從高點回撤超過 10% 即停止交易 |
| `max_daily_loss_multiplier` | 2× | 單日虧損達 2 × `max_loss_per_trade` 後停止交易 |
| `max_loss_per_trade` | $500 | 設定日損失上限的基準規模 |
| `max_concurrent_positions` | 1 | 同時只持有 1 個鐵禿鷹部位 |
| 日曆過濾 | FOMC/CPI/NFP | 2025–2026 硬編碼備援；若提供 FRED API 金鑰則優先使用 FRED |

**`ALPACA_PAPER=true` 在客戶端初始化時強制驗證** — 程式庫拒絕在真實帳戶上運行。兩個範例腳本在提交任何委託單前均會要求使用者確認。

---

## 模組參考

### `alpaca_options.client`

| 函式 | 說明 |
|------|------|
| `get_clients() → (TradingClient, OptionHistoricalDataClient)` | 回傳已快取的模擬模式 Alpaca 客戶端對。強制驗證 `ALPACA_PAPER=true`。 |

---

### `alpaca_options.contracts`

| 函式 | 說明 |
|------|------|
| `get_option_contracts(underlying_symbol, expiration_gte, expiration_lte, strike_gte, strike_lte, contract_type, limit) → List[OptionContract]` | 過濾並回傳選擇權合約清單。 |
| `find_atm_call(underlying_symbol, expiration, underlying_price, strike_window) → OptionContract` | 回傳最接近價平的買權合約。 |
| `find_atm_put(underlying_symbol, expiration, underlying_price, strike_window) → OptionContract` | 回傳最接近價平的賣權合約。 |

---

### `alpaca_options.quotes`

| 函式 | 說明 |
|------|------|
| `get_latest_quote(symbol) → Quote` | 擷取 OCC 選擇權代碼的最新全國最優買賣報價（NBBO）。 |
| `get_option_bars(symbol, timeframe, start, end, limit) → List[Bar]` | 擷取 OHLCV K 線資料。 |
| `midpoint(quote) → float` | 買賣價中間價；可處理單側為零的報價。 |

---

### `alpaca_options.orders`

| 函式 | 說明 |
|------|------|
| `buy_to_open_market(symbol, qty) → Order` | 市價開多（BTO）。 |
| `buy_to_open_limit(symbol, qty, limit_price) → Order` | 限價開多（BTO）。 |
| `sell_to_close_market(symbol, qty) → Order` | 市價平多（STC）。 |
| `sell_to_close_limit(symbol, qty, limit_price) → Order` | 限價平多（STC）。 |
| `bull_call_spread(long_symbol, short_symbol, qty, net_debit) → Order` | 多腳牛市買權價差委託單。 |
| `straddle(call_symbol, put_symbol, qty, net_debit) → Order` | 多腳多頭跨式委託單。 |

---

### `alpaca_options.positions`

| 函式 | 說明 |
|------|------|
| `list_option_positions() → pd.DataFrame` | 列出含損益欄位的未平倉選擇權部位。 |
| `close_all_options(cancel_orders) → List[ClosePositionResponse]` | 平倉所有部位。 |

---

### `alpaca_options.data.calendar`

| 函式 | 說明 |
|------|------|
| `get_event_days(start, end) → set[date]` | 回傳指定區間內的 FOMC、CPI、NFP 日期集合。若設有 `FRED_API_KEY` 環境變數則使用 FRED API；否則使用硬編碼備援日曆。 |

---

### `alpaca_options.data.vix`

| 函式 | 說明 |
|------|------|
| `get_current_vix() → float` | 透過 Alpaca VIXY 代理或 yfinance 備援取得當前 VIX 指數。 |
| `is_high_vol_regime(vix, threshold) → bool` | VIX ≥ 閾值（預設 25）時回傳 True。 |

---

### `alpaca_options.risk.manager`

| 方法 | 說明 |
|------|------|
| `RiskManager(vix_threshold, max_daily_loss_multiplier, max_drawdown_pct, max_concurrent_positions, max_loss_per_trade)` | 建構子——所有參數均有合理預設值。 |
| `reset_day(account_value)` | 每個交易日開始時重置日內計數器。 |
| `check_entry_allowed(account_value, vix, today, calendar_events) → (bool, str)` | 檢查全部 5 條防護規則。回傳 `(True, "OK")` 或 `(False, 原因)`。 |
| `open_position()` | 遞增未平倉部位計數器。 |
| `record_trade_result(pnl)` | 交易平倉後更新狀態。 |
| `summary() → dict` | 回傳風險狀態快照。 |

---

### `alpaca_options.strategies.iron_condor_0dte`

| 項目 | 說明 |
|------|------|
| `IronCondorConfig` | 包含所有策略參數的資料類別（dataclass）。 |
| `CondorLegs` | 含 4 個 OCC 代碼與履約價的不可變資料類別。 |
| `CondorPosition` | 可變的即時部位物件（每次 tick 更新 current_value）。 |
| `ExitDecision` | 列舉：HOLD / CLOSE_PROFIT / CLOSE_STOP / CLOSE_DELTA_BREACH / CLOSE_TIME。 |
| `IronCondor0DTE.should_enter(now, vix) → bool` | 時間窗口檢查（美東時間 10:00–14:00）。 |
| `IronCondor0DTE.build_condor(underlying) → CondorLegs \| None` | 擷取合約鏈、計算 Delta 履約價、驗證權利金，回傳腳位組合或 None。 |
| `IronCondor0DTE.enter(legs) → order_id` | 提交多腳限價開倉委託單。 |
| `IronCondor0DTE.monitor(position, now) → ExitDecision` | 檢查獲利目標、停損、Delta 突破及時間條件。 |
| `IronCondor0DTE.exit(position, reason) → order_id` | 提交多腳市價平倉委託單。 |

---

### `alpaca_options.backtest.replay`

| 項目 | 說明 |
|------|------|
| `BacktestEngine(initial_equity, vix_override)` | 建構子。 |
| `BacktestEngine.run(start, end, strategy, risk) → BacktestResults` | 在指定日期區間模擬策略。擷取 SPY 1 分鐘 K 線；優先使用真實選擇權 K 線，不足時以 Black-Scholes 補全。 |
| `BacktestResults.save_equity_curve_plot(path)` | 儲存含資產曲線與每日損益柱狀圖的 PNG 檔。 |

---

### `alpaca_options.live.runner`

| 項目 | 說明 |
|------|------|
| `LiveRunner(dry_run, strategy, risk, poll_interval)` | 建構子。`dry_run=True` 為預設值。 |
| `LiveRunner.run()` | 主循環（阻塞式執行）。交易時段每 60 秒輪詢一次。支援 SIGINT 優雅退出。 |

---

### `alpaca_options.utils.black_scholes`

| 函式 | 說明 |
|------|------|
| `bs_price(S, K, T, r, sigma, is_call) → float` | Black-Scholes 選擇權理論價格。 |
| `bs_delta(S, K, T, r, sigma, is_call) → float` | Black-Scholes Delta 值。 |
| `strike_for_delta(S, T, r, sigma, delta, is_call) → float` | 給定目標 Delta 的封閉解履約價。 |
| `norm_cdf(x) → float` | 標準常態累積分布函數。 |
| `norm_ppf(p) → float` | 標準常態反累積分布函數（無需 scipy）。 |

---

## 專案目錄結構

```
alpaca-options/
├── .env                             # API 憑證（切勿提交至版本控制！）
├── pyproject.toml
├── src/
│   └── alpaca_options/
│       ├── __init__.py
│       ├── client.py                # 客戶端工廠（paper=True 防護）
│       ├── contracts.py             # 合約探索 + 價平輔助函式
│       ├── quotes.py                # 最新報價 + 歷史 K 線
│       ├── orders.py                # 單腳 + 多腳委託單提交
│       ├── positions.py             # 部位列表 + 損益 DataFrame
│       ├── utils/
│       │   └── black_scholes.py     # BS 定價、Delta、Delta 反推履約價
│       ├── data/
│       │   ├── calendar.py          # FOMC/CPI/NFP 事件日曆
│       │   └── vix.py               # VIX 擷取 + 市場環境偵測
│       ├── risk/
│       │   └── manager.py           # RiskManager——5 條進場防護規則
│       ├── strategies/
│       │   └── iron_condor_0dte.py  # IronCondor0DTE 策略類別
│       ├── backtest/
│       │   ├── replay.py            # BacktestEngine + BacktestResults
│       │   └── _occ.py              # OCC 代碼產生器
│       └── live/
│           └── runner.py            # LiveRunner（空跑 + 即時模式）
├── examples/
│   ├── buy_atm_call.py
│   ├── vertical_spread.py
│   ├── risk_manager_demo.py
│   ├── backtest_90_days.py
│   └── live_dry_run.py
└── tests/
    ├── conftest.py
    ├── test_risk_manager.py
    ├── test_iron_condor.py
    └── test_backtest.py
```

---

## 日誌記錄

```python
import logging
logging.getLogger("alpaca_options").setLevel(logging.INFO)
```

---

## FRED API 金鑰（選用，提升事件日曆精確度）

```bash
# 在以下網址免費申請金鑰：https://fred.stlouisfed.org/docs/api/api_key.html
# 加入 .env：
FRED_API_KEY=your_key_here
```

---

## 安全提醒

- `ALPACA_PAPER=true` 在客戶端初始化時**強制驗證**。
- 兩個範例腳本在提交任何委託單前**均會要求確認**。
- 回測使用 K 線中間價成交——實際交易結果將更差。
- 選擇權交易涉及重大風險，即使在模擬環境下亦然。
