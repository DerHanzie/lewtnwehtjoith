#!/opt/venv/bin/python3
"""
Sovereign Crypto Architect V2.2 - Python Quant Engine
This script is called by n8n's Execute Command node.
Input: JSON via stdin (coin, timeframe)
Output: JSON via stdout (analysis results)

NOTE: Using pure pandas for indicators (no pandas-ta/numba dependency)
"""
import sys
import json
import ccxt
import pandas as pd
import mplfinance as mpf
import io
import base64

# === INDICATOR FUNCTIONS (Pure Pandas) ===
def calc_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    """Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def main():
    # 1. Read input from stdin (n8n passes JSON)
    try:
        input_data = json.loads(sys.stdin.read())
    except:
        input_data = {"Coin": "BTC", "Timeframe": "4h"}
    
    coin = input_data.get('Coin', 'BTC')
    timeframe = input_data.get('Timeframe', '4h')
    coin_pair = f"{coin}/USDT"
    
    # 2. Data Acquisition
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(coin_pair, timeframe, limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # BTC Correlation
    btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=500)
    df_btc = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 3. Indicators (Pure Pandas - No pandas-ta)
    df['EMA_50'] = calc_ema(df['close'], 50)
    df['EMA_200'] = calc_ema(df['close'], 200)
    df['RSI_14'] = calc_rsi(df['close'], 14)
    df['ATR_14'] = calc_atr(df, 14)
    df_btc['EMA_200'] = calc_ema(df_btc['close'], 200)
    
    # 4. Logic
    current_price = float(df['close'].iloc[-1])
    ema_50 = float(df['EMA_50'].iloc[-1])
    ema_200 = float(df['EMA_200'].iloc[-1])
    rsi_value = float(df['RSI_14'].iloc[-1])
    atr_value = float(df['ATR_14'].iloc[-1])
    btc_price = float(df_btc['close'].iloc[-1])
    btc_ema_200 = float(df_btc['EMA_200'].iloc[-1])
    
    ema_spread = (ema_50 - ema_200) / ema_200
    bullish_regime = False
    trend_strength = "NEUTRAL"
    dynamic_rsi_threshold = 30
    
    if current_price > ema_200 and ema_50 > ema_200:
        bullish_regime = True
        if ema_spread > 0.02:
            trend_strength = "STRONG_BULL"
            dynamic_rsi_threshold = 45
        else:
            trend_strength = "WEAK_BULL"
            dynamic_rsi_threshold = 35
    
    # Opus Refinement: Dynamic RSI Exit
    rsi_exit_threshold = 80 if trend_strength == "STRONG_BULL" else 75
    
    btc_bullish = btc_price > btc_ema_200
    signal = "WAIT"
    unwind_position = False
    reason = []
    
    if current_price < ema_200:
        unwind_position = True
        reason.append("Trend Broken")
    elif rsi_value > rsi_exit_threshold:
        unwind_position = True
        reason.append(f"RSI Overextended (>{rsi_exit_threshold})")
    
    if unwind_position:
        signal = "SELL"
    else:
        if bullish_regime and btc_bullish:
            if rsi_value < dynamic_rsi_threshold:
                signal = "BUY_CANDIDATE"
                reason.append(f"Trend {trend_strength}, RSI OK")
            else:
                reason.append("RSI too high")
        else:
            reason.append("Trend Filter Failed")
    
    # 5. Visuals
    buf = io.BytesIO()
    mc = mpf.make_marketcolors(up='#2ebd85', down='#f6465d', volume='in')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', rc={'font.size': 12})
    add_plots = [
        mpf.make_addplot(df['EMA_50'].tail(80), color='orange', width=1.5),
        mpf.make_addplot(df['EMA_200'].tail(80), color='blue', width=2.0)
    ]
    mpf.plot(df.tail(80), type='candle', volume=True, addplot=add_plots, style=s, figsize=(10, 8), savefig=dict(fname=buf, format='png'))
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # 6. Output JSON
    result = {
        "coin": coin,
        "price": current_price,
        "indicators": {
            "rsi": rsi_value,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "atr": atr_value
        },
        "market_context": {
            "trend_strength": trend_strength,
            "btc_bullish": btc_bullish,
            "rsi_exit_threshold": rsi_exit_threshold
        },
        "logic_engine": {
            "signal": signal,
            "unwind_position": unwind_position,
            "reason": "; ".join(reason)
        },
        "image_base64": image_base64
    }
    
    # Output to stdout for n8n to capture
    print(json.dumps(result))

if __name__ == "__main__":
    main()
