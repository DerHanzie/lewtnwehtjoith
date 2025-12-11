#!/opt/venv/bin/python3
"""
Sovereign Crypto Architect V2.5 - Python Quant Engine
Enhanced with Multi-Timeframe Confirmation, Score Breakdown, and Fixed SELL Logic

Input: JSON via stdin (coin, timeframe)
Output: JSON via stdout (analysis results)

Philosophy: Quality over quantity. One good trade per day > many mediocre ones.
High confidence = near-certain win.
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

def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD with histogram for momentum confirmation"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def analyze_volume(df, lookback=20):
    """
    Determine if volume is accumulation or distribution.
    This is CRITICAL for avoiding bull traps.
    """
    avg_vol = df['volume'].rolling(lookback).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    
    # Use last 5 candles to determine price direction during volume
    recent = df.tail(5)
    price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
    
    vol_type = "neutral"
    strength = "weak"
    
    if volume_ratio > 1.5:
        strength = "strong"
        if price_change > 0.01:  # Price up with high volume = buying
            vol_type = "accumulation"
        elif price_change < -0.01:  # Price down with high volume = selling
            vol_type = "distribution"
    elif volume_ratio > 1.2:
        strength = "moderate"
        if price_change > 0.005:
            vol_type = "accumulation"
        elif price_change < -0.005:
            vol_type = "distribution"
    
    return {
        "type": vol_type,
        "ratio": round(volume_ratio, 2),
        "strength": strength,
        "price_change_pct": round(price_change * 100, 2)
    }

def safe_float(value, default=0.0):
    """Safely convert to float, handling NaN values"""
    try:
        result = float(value)
        if pd.isna(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def analyze_timeframe(exchange, coin_pair, timeframe):
    """
    Analyze a single timeframe and return trend status.
    Returns dict with bullish status and key indicators.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(coin_pair, timeframe, limit=250)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df['EMA_50'] = calc_ema(df['close'], 50)
        df['EMA_200'] = calc_ema(df['close'], 200)
        
        current_price = safe_float(df['close'].iloc[-1])
        ema_50 = safe_float(df['EMA_50'].iloc[-1])
        ema_200 = safe_float(df['EMA_200'].iloc[-1])
        
        is_bullish = current_price > ema_200 and ema_50 > ema_200
        ema_spread = (ema_50 - ema_200) / ema_200 if ema_200 > 0 else 0
        
        return {
            "timeframe": timeframe,
            "is_bullish": is_bullish,
            "price": current_price,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "ema_spread_pct": round(ema_spread * 100, 2)
        }
    except Exception as e:
        return {
            "timeframe": timeframe,
            "is_bullish": False,
            "error": str(e)
        }


def check_multi_timeframe_alignment(exchange, coin_pair, primary_tf='4h'):
    """
    Check trend alignment across multiple timeframes.
    Returns alignment score and details.
    
    Timeframe hierarchy:
    - 1h: Short-term momentum
    - 4h: Primary trading timeframe
    - 1d: Major trend direction
    """
    timeframes_to_check = ['1h', '4h', '1d']
    
    # Remove primary if already in list to avoid duplicate
    if primary_tf in timeframes_to_check:
        timeframes_to_check.remove(primary_tf)
    timeframes_to_check.insert(0, primary_tf)
    
    results = []
    bullish_count = 0
    
    for tf in timeframes_to_check[:3]:  # Max 3 timeframes
        analysis = analyze_timeframe(exchange, coin_pair, tf)
        results.append(analysis)
        if analysis.get("is_bullish", False):
            bullish_count += 1
    
    # Calculate alignment
    total_checked = len(results)
    alignment_score = bullish_count / total_checked if total_checked > 0 else 0
    
    # Determine overall alignment status
    if bullish_count == total_checked:
        alignment_status = "FULL_ALIGNMENT"
    elif bullish_count >= 2:
        alignment_status = "PARTIAL_ALIGNMENT"
    elif bullish_count == 1:
        alignment_status = "WEAK_ALIGNMENT"
    else:
        alignment_status = "NO_ALIGNMENT"
    
    return {
        "alignment_status": alignment_status,
        "alignment_score": round(alignment_score, 2),
        "bullish_timeframes": bullish_count,
        "total_timeframes": total_checked,
        "details": results
    }


def main():
    try:
        # 1. Read input from stdin (n8n passes JSON)
        try:
            input_data = json.loads(sys.stdin.read())
        except:
            input_data = {"Coin": "BTC", "Timeframe": "4h"}
        
        coin = input_data.get('Coin', 'BTC')
        timeframe_raw = input_data.get('Timeframe', '4h')
        risk_profile_raw = input_data.get('RiskProfile', 'Standard')
        
        # Robust extraction of values from emoji-prefixed Notion strings
        import re
        
        # Timeframe: look for patterns like 1h, 4h, 1d, 15m, etc.
        timeframe_match = re.search(r'\b(\d+[mhdwM])\b', timeframe_raw or '')
        timeframe = timeframe_match.group(1) if timeframe_match else '4h'
        
        # Risk Profile: look for known keywords
        risk_keywords = ['Conservative', 'Standard', 'Aggressive']
        risk_profile = 'Standard'  # default
        for keyword in risk_keywords:
            if keyword.lower() in (risk_profile_raw or '').lower():
                risk_profile = keyword
                break
        
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
        df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
        df_btc.set_index('timestamp', inplace=True)
        
        # 3. Indicators (Pure Pandas)
        df['EMA_50'] = calc_ema(df['close'], 50)
        df['EMA_200'] = calc_ema(df['close'], 200)
        df['RSI_14'] = calc_rsi(df['close'], 14)
        df['ATR_14'] = calc_atr(df, 14)
        
        # MACD for momentum confirmation
        macd_line, signal_line, histogram = calc_macd(df['close'])
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Hist'] = histogram
        
        df_btc['EMA_200'] = calc_ema(df_btc['close'], 200)
        
        # 4. Extract current values
        current_price = safe_float(df['close'].iloc[-1])
        ema_50 = safe_float(df['EMA_50'].iloc[-1])
        ema_200 = safe_float(df['EMA_200'].iloc[-1])
        rsi_value = safe_float(df['RSI_14'].iloc[-1], 50.0)
        atr_value = safe_float(df['ATR_14'].iloc[-1])
        btc_price = safe_float(df_btc['close'].iloc[-1])
        btc_ema_200 = safe_float(df_btc['EMA_200'].iloc[-1])
        
        # MACD values
        macd_hist_current = safe_float(df['MACD_Hist'].iloc[-1])
        macd_hist_prev = safe_float(df['MACD_Hist'].iloc[-2])
        macd_line_val = safe_float(df['MACD'].iloc[-1])
        macd_signal_val = safe_float(df['MACD_Signal'].iloc[-1])
        
        # Volume analysis
        volume_analysis = analyze_volume(df)
        
        # 5. Multi-Timeframe Analysis (NEW in V2.5)
        mtf_analysis = check_multi_timeframe_alignment(exchange, coin_pair, timeframe)
        
        # 6. Trend Analysis
        if ema_200 == 0:
            ema_spread = 0
        else:
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
        
        rsi_exit_threshold = 80 if trend_strength == "STRONG_BULL" else 75
        btc_bullish = btc_price > btc_ema_200
        
        # 7. MACD Momentum Analysis
        macd_bullish = macd_hist_current > 0
        macd_accelerating = macd_hist_current > macd_hist_prev
        macd_crossover = macd_line_val > macd_signal_val and macd_hist_prev <= 0
        
        # 8. Signal Logic (Enhanced with MTF)
        signal = "WAIT"
        unwind_position = False
        reason = []
        quality_flags = []
        score_breakdown = []  # NEW: Detailed scoring breakdown
        
        # EXIT CHECKS
        if current_price < ema_200:
            unwind_position = True
            reason.append("Trend Broken (Price < EMA200)")
        elif rsi_value > rsi_exit_threshold:
            unwind_position = True
            reason.append(f"RSI Overextended (>{rsi_exit_threshold})")
        
        if unwind_position:
            signal = "SELL"
        else:
            # ENTRY EVALUATION
            trend_ok = bullish_regime and btc_bullish
            rsi_ok = rsi_value < dynamic_rsi_threshold
            
            # Volume must not be distribution
            volume_ok = volume_analysis["type"] != "distribution"
            volume_bonus = volume_analysis["type"] == "accumulation"
            
            # MACD momentum check
            momentum_ok = macd_bullish or macd_accelerating
            
            # NEW: Multi-timeframe alignment check
            mtf_ok = mtf_analysis["alignment_status"] in ["FULL_ALIGNMENT", "PARTIAL_ALIGNMENT"]
            
            if trend_ok and rsi_ok:
                if volume_analysis["type"] == "distribution":
                    signal = "WAIT"
                    reason.append("Volume shows distribution (selling pressure)")
                    quality_flags.append("VOLUME_WARNING")
                elif not mtf_ok:
                    signal = "WAIT"
                    reason.append(f"Multi-TF not aligned ({mtf_analysis['alignment_status']})")
                    quality_flags.append("MTF_MISALIGNMENT")
                elif not momentum_ok and rsi_value > 40:
                    signal = "WAIT"
                    reason.append("MACD momentum not confirming, RSI not oversold enough")
                else:
                    signal = "BUY_CANDIDATE"
                    reason.append(f"Trend: {trend_strength}")
                    
                    # Track quality for confidence scoring
                    if rsi_value >= 25 and rsi_value <= 35:
                        quality_flags.append("OPTIMAL_RSI")
                    if volume_bonus:
                        quality_flags.append("ACCUMULATION")
                    if macd_accelerating:
                        quality_flags.append("MOMENTUM_ACCELERATING")
                    if macd_crossover:
                        quality_flags.append("MACD_CROSSOVER")
                    if trend_strength == "STRONG_BULL":
                        quality_flags.append("STRONG_TREND")
                    if mtf_analysis["alignment_status"] == "FULL_ALIGNMENT":
                        quality_flags.append("MTF_FULL_ALIGNMENT")
                    
                    # Momentum breakout exception
                    if rsi_value > dynamic_rsi_threshold and rsi_value < 60:
                        if volume_analysis["ratio"] > 2.0 and macd_crossover:
                            signal = "BUY_CANDIDATE"
                            reason.append("Momentum breakout (high volume + MACD crossover)")
                            quality_flags.append("BREAKOUT_ENTRY")
            else:
                if not bullish_regime:
                    reason.append("Price below trend (EMA filter)")
                if not btc_bullish:
                    reason.append("BTC bearish (correlation filter)")
        
        # 9. Score Breakdown Calculation (NEW in V2.5)
        # Always calculate for transparency
        trend_score = 40 if trend_strength == "STRONG_BULL" else (20 if trend_strength == "WEAK_BULL" else 0)
        rsi_score = 25 if (25 <= rsi_value <= 35) else (15 if (20 <= rsi_value <= 45) else (10 if rsi_value < 20 else 0))
        volume_score = 20 if volume_analysis["type"] == "accumulation" else (-25 if volume_analysis["type"] == "distribution" else 0)
        macd_score = 15 if macd_crossover else (10 if (macd_accelerating and macd_bullish) else (5 if macd_bullish else 0))
        btc_penalty = -15 if not btc_bullish else 0
        mtf_bonus = 10 if mtf_analysis["alignment_status"] == "FULL_ALIGNMENT" else (5 if mtf_analysis["alignment_status"] == "PARTIAL_ALIGNMENT" else 0)
        
        raw_score = trend_score + rsi_score + volume_score + macd_score + btc_penalty + mtf_bonus
        
        score_breakdown = [
            f"Trend: {'+' if trend_score >= 0 else ''}{trend_score} ({trend_strength})",
            f"RSI: {'+' if rsi_score >= 0 else ''}{rsi_score} ({rsi_value:.1f})",
            f"Volume: {'+' if volume_score >= 0 else ''}{volume_score} ({volume_analysis['type']})",
            f"MACD: {'+' if macd_score >= 0 else ''}{macd_score}",
            f"MTF: {'+' if mtf_bonus >= 0 else ''}{mtf_bonus} ({mtf_analysis['alignment_status']})",
        ]
        if btc_penalty != 0:
            score_breakdown.append(f"BTC: {btc_penalty}")
        
        score_breakdown_str = " | ".join(score_breakdown)
        
        # 10. ATR-Based Risk/Reward
        # Only calculate meaningful values for BUY signals
        if signal == "SELL" or signal == "WAIT":
            # For SELL/WAIT: set to 0 (no targets needed)
            suggested_stop_loss = 0
            suggested_target = 0
        else:
            atr_stop_distance = 2 * atr_value
            suggested_stop_loss = current_price - atr_stop_distance
            risk = current_price - suggested_stop_loss
            suggested_target = current_price + (risk * 2)  # 2:1 R:R minimum
        
        # 11. Visuals
        buf = io.BytesIO()
        mc = mpf.make_marketcolors(up='#2ebd85', down='#f6465d', volume='in')
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', rc={'font.size': 12})
        
        # Enhanced plot with MACD panel
        add_plots = [
            mpf.make_addplot(df['EMA_50'].tail(80), color='orange', width=1.5),
            mpf.make_addplot(df['EMA_200'].tail(80), color='blue', width=2.0),
            mpf.make_addplot(df['MACD_Hist'].tail(80), type='bar', panel=2, color='dimgray', secondary_y=False)
        ]
        mpf.plot(df.tail(80), type='candle', volume=True, addplot=add_plots, style=s, 
                 figsize=(10, 10), panel_ratios=(4, 1, 1), savefig=dict(fname=buf, format='png'))
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # 12. Output JSON (Enhanced V2.5)
        result = {
            "coin": coin,
            "price": round(current_price, 4),
            "risk_profile": risk_profile,
            "indicators": {
                "rsi": round(rsi_value, 2),
                "ema_50": round(ema_50, 4),
                "ema_200": round(ema_200, 4),
                "atr": round(atr_value, 6),
                "macd_histogram": round(macd_hist_current, 6),
                "macd_accelerating": macd_accelerating
            },
            "market_context": {
                "trend_strength": trend_strength,
                "btc_bullish": btc_bullish,
                "rsi_exit_threshold": rsi_exit_threshold,
                "ema_spread_pct": round(ema_spread * 100, 2)
            },
            "volume_analysis": volume_analysis,
            "momentum": {
                "macd_bullish": macd_bullish,
                "macd_accelerating": macd_accelerating,
                "macd_crossover": macd_crossover
            },
            "multi_timeframe": mtf_analysis,  # NEW in V2.5
            "risk_management": {
                "atr_value": round(atr_value, 6),
                "suggested_stop_loss": round(suggested_stop_loss, 4),
                "suggested_target": round(suggested_target, 4),
                "risk_reward_ratio": 2.0 if signal == "BUY_CANDIDATE" else 0
            },
            "logic_engine": {
                "signal": signal,
                "unwind_position": unwind_position,
                "reason": "; ".join(reason) if reason else "No specific conditions triggered",
                "quality_flags": quality_flags,
                "score_breakdown": score_breakdown_str,  # NEW in V2.5
                "raw_score": max(0, min(100, raw_score))  # NEW in V2.5
            },
            "image_base64": image_base64
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "error": True,
            "message": str(e),
            "coin": input_data.get('Coin', 'UNKNOWN') if 'input_data' in dir() else 'UNKNOWN',
            "signal": "ERROR",
            "reason": f"Script error: {str(e)}"
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
