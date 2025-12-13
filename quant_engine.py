#!/opt/venv/bin/python3
"""
Sovereign Crypto Architect V2.6 - Python Quant Engine
Strategic Upgrades: Bear Trap Protection, Volatility Filter, ADX, Momentum Override, Funding Rates

Input: JSON via stdin (coin, timeframe)
Output: JSON via stdout (analysis results)

Philosophy: Quality over quantity. Protect capital first, then seek gains.
"""
import sys
import json
import ccxt
import pandas as pd
import mplfinance as mpf
import io
import base64
import pytz
from datetime import datetime

# === DATETIME CONFIGURATION ===
# Project timezone: Europe/Amsterdam (CET/CEST)
TIMEZONE = pytz.timezone('Europe/Amsterdam')
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M'
DATETIME_FORMAT_FULL = '%Y-%m-%d %H:%M:%S %Z'

def get_current_time():
    """Get current time in Amsterdam timezone."""
    return datetime.now(TIMEZONE)

def format_timestamp(timestamp_ms, format_str=DATETIME_FORMAT):
    """Convert millisecond timestamp to formatted string in Amsterdam timezone."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
    dt_local = dt.astimezone(TIMEZONE)
    return dt_local.strftime(format_str)

def to_amsterdam_tz(dt):
    """Convert any datetime to Amsterdam timezone."""
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(TIMEZONE)

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

def calc_adx(df, period=14):
    """
    Average Directional Index (ADX) - Measures trend strength
    ADX > 25 = Trending market
    ADX < 20 = Ranging/dead market
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate directional movement
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

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
        if price_change > 0.01:
            vol_type = "accumulation"
        elif price_change < -0.01:
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
    """Analyze a single timeframe and return trend status."""
    try:
        ohlcv = exchange.fetch_ohlcv(coin_pair, timeframe, limit=250)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert to timezone-aware datetime (UTC -> Europe/Amsterdam)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Amsterdam')
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
    """Check trend alignment across multiple timeframes."""
    timeframes_to_check = ['1h', '4h', '1d']
    
    if primary_tf in timeframes_to_check:
        timeframes_to_check.remove(primary_tf)
    timeframes_to_check.insert(0, primary_tf)
    
    results = []
    bullish_count = 0
    
    for tf in timeframes_to_check[:3]:
        analysis = analyze_timeframe(exchange, coin_pair, tf)
        results.append(analysis)
        if analysis.get("is_bullish", False):
            bullish_count += 1
    
    total_checked = len(results)
    alignment_score = bullish_count / total_checked if total_checked > 0 else 0
    
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


def fetch_funding_rate(exchange, coin_pair):
    """
    Fetch funding rate for perpetual futures.
    High positive funding = overbought (longs paying shorts)
    High negative funding = oversold (shorts paying longs)
    """
    try:
        # Convert spot pair to futures format
        symbol = coin_pair.replace('/', '')
        funding = exchange.fetch_funding_rate(symbol)
        rate = funding.get('fundingRate', 0) or 0
        return {
            "rate": round(rate * 100, 4),  # Convert to percentage
            "warning": abs(rate) > 0.0005,  # > 0.05% is notable
            "extreme": abs(rate) > 0.001,   # > 0.1% is extreme
            "direction": "LONGS_PAYING" if rate > 0 else "SHORTS_PAYING" if rate < 0 else "NEUTRAL"
        }
    except Exception as e:
        # Fallback if funding rate not available (spot only exchange)
        return {
            "rate": 0,
            "warning": False,
            "extreme": False,
            "direction": "UNAVAILABLE",
            "error": str(e)
        }


def check_bear_trap_protection(df, ema_200, buffer_pct=0.01):
    """
    V2.6 Bear Trap Protection:
    Only trigger SELL if the CLOSED candle is below EMA200 * (1 - buffer)
    This prevents selling on wicks/liquidity grabs.
    
    Args:
        df: DataFrame with OHLCV data
        ema_200: Current EMA200 value
        buffer_pct: Buffer percentage (default 1%)
    
    Returns:
        dict with breakdown confirmation status
    """
    # Get the PREVIOUS closed candle (not current incomplete one)
    # In our case, we already exclude incomplete candles, so use last
    last_close = safe_float(df['close'].iloc[-1])
    prev_close = safe_float(df['close'].iloc[-2])
    
    # Calculate the buffered threshold
    breakdown_threshold = ema_200 * (1 - buffer_pct)
    
    # Check if BOTH last two candles closed below threshold (confirmation)
    last_below = last_close < breakdown_threshold
    prev_below = prev_close < breakdown_threshold
    
    # Single candle below = warning, two candles = confirmed breakdown
    confirmed_breakdown = last_below and prev_below
    warning_breakdown = last_below and not prev_below
    
    return {
        "confirmed_breakdown": confirmed_breakdown,
        "warning_breakdown": warning_breakdown,
        "threshold": round(breakdown_threshold, 4),
        "last_close": round(last_close, 4),
        "prev_close": round(prev_close, 4),
        "buffer_pct": buffer_pct * 100
    }


def check_volatility_filter(df, atr_period=14, multiplier=2.0):
    """
    V2.6 Volatility Filter:
    Block trades when ATR is abnormally high (chaos/crash conditions).
    
    Args:
        df: DataFrame with OHLCV data
        atr_period: Period for ATR calculation
        multiplier: How many times average ATR is considered "chaos"
    
    Returns:
        dict with volatility status
    """
    atr_series = calc_atr(df, atr_period)
    current_atr = safe_float(atr_series.iloc[-1])
    avg_atr = safe_float(atr_series.rolling(50).mean().iloc[-1])
    
    if avg_atr == 0:
        atr_ratio = 1.0
    else:
        atr_ratio = current_atr / avg_atr
    
    is_chaotic = atr_ratio > multiplier
    is_elevated = atr_ratio > 1.5
    
    # Determine market volatility regime
    if atr_ratio > 3.0:
        regime = "EXTREME_VOLATILITY"
    elif atr_ratio > 2.0:
        regime = "HIGH_VOLATILITY"
    elif atr_ratio > 1.5:
        regime = "ELEVATED_VOLATILITY"
    else:
        regime = "NORMAL"
    
    return {
        "current_atr": round(current_atr, 6),
        "average_atr": round(avg_atr, 6),
        "atr_ratio": round(atr_ratio, 2),
        "regime": regime,
        "is_chaotic": is_chaotic,
        "is_elevated": is_elevated,
        "block_trades": is_chaotic,  # Block if chaotic
        "reduce_confidence": is_elevated  # Reduce confidence if elevated
    }


def check_momentum_override(df, ema_spread, volume_analysis, macd_crossover, rsi_value):
    """
    V2.6 Momentum Override:
    In parabolic/strong trend conditions, relax RSI entry requirements.
    
    This catches big moves that traditional RSI rules would miss.
    """
    # Check for parabolic conditions
    is_parabolic = ema_spread > 0.05  # > 5% spread between EMA50 and EMA200
    
    # Check for momentum breakout
    high_volume = volume_analysis.get("ratio", 0) > 2.0
    momentum_surge = macd_crossover and high_volume
    
    # Determine if we should override RSI rules
    override_active = False
    max_rsi_override = 45  # Default
    reason = ""
    
    if is_parabolic:
        override_active = True
        max_rsi_override = 55  # Allow entries up to RSI 55 in parabolic moves
        reason = f"Parabolic trend (EMA spread {ema_spread*100:.1f}%)"
    elif momentum_surge:
        override_active = True
        max_rsi_override = 50  # Allow entries up to RSI 50 on momentum surges
        reason = f"Momentum surge (Volume {volume_analysis.get('ratio', 0):.1f}x + MACD crossover)"
    
    # Still reject if RSI is truly overbought
    if rsi_value > 65:
        override_active = False
        reason = f"RSI too high ({rsi_value:.1f}) even for override"
    
    return {
        "override_active": override_active,
        "max_rsi_allowed": max_rsi_override,
        "is_parabolic": is_parabolic,
        "is_momentum_surge": momentum_surge,
        "reason": reason
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
        
        timeframe_match = re.search(r'\b(\d+[mhdwM])\b', timeframe_raw or '')
        timeframe = timeframe_match.group(1) if timeframe_match else '4h'
        
        risk_keywords = ['Conservative', 'Standard', 'Aggressive']
        risk_profile = 'Standard'
        for keyword in risk_keywords:
            if keyword.lower() in (risk_profile_raw or '').lower():
                risk_profile = keyword
                break
        
        coin_pair = f"{coin}/USDT"
        
        # 2. Data Acquisition
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(coin_pair, timeframe, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert to timezone-aware datetime (UTC -> Europe/Amsterdam)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Amsterdam')
        df.set_index('timestamp', inplace=True)
        
        # BTC Correlation
        btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=500)
        df_btc = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert to timezone-aware datetime (UTC -> Europe/Amsterdam)
        df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms', utc=True)
        df_btc['timestamp'] = df_btc['timestamp'].dt.tz_convert('Europe/Amsterdam')
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
        
        # V2.6: ADX for trend strength
        adx_series, plus_di, minus_di = calc_adx(df)
        df['ADX'] = adx_series
        
        df_btc['EMA_200'] = calc_ema(df_btc['close'], 200)
        
        # 4. Extract current values
        current_price = safe_float(df['close'].iloc[-1])
        ema_50 = safe_float(df['EMA_50'].iloc[-1])
        ema_200 = safe_float(df['EMA_200'].iloc[-1])
        rsi_value = safe_float(df['RSI_14'].iloc[-1], 50.0)
        atr_value = safe_float(df['ATR_14'].iloc[-1])
        adx_value = safe_float(df['ADX'].iloc[-1], 20.0)
        btc_price = safe_float(df_btc['close'].iloc[-1])
        btc_ema_200 = safe_float(df_btc['EMA_200'].iloc[-1])
        
        # MACD values
        macd_hist_current = safe_float(df['MACD_Hist'].iloc[-1])
        macd_hist_prev = safe_float(df['MACD_Hist'].iloc[-2])
        macd_line_val = safe_float(df['MACD'].iloc[-1])
        macd_signal_val = safe_float(df['MACD_Signal'].iloc[-1])
        
        # Volume analysis
        volume_analysis = analyze_volume(df)
        
        # 5. V2.6 Strategic Checks
        mtf_analysis = check_multi_timeframe_alignment(exchange, coin_pair, timeframe)
        volatility_check = check_volatility_filter(df)
        bear_trap_check = check_bear_trap_protection(df, ema_200)
        
        # Funding rate (may fail on spot-only)
        funding_rate = fetch_funding_rate(exchange, coin_pair)
        
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
        
        # V2.6: Momentum Override Check
        momentum_override = check_momentum_override(
            df, ema_spread, volume_analysis, macd_crossover, rsi_value
        )
        
        # Apply momentum override to RSI threshold
        if momentum_override["override_active"]:
            dynamic_rsi_threshold = momentum_override["max_rsi_allowed"]
        
        # V2.6: ADX Market Regime
        adx_regime = "TRENDING" if adx_value > 25 else ("WEAK_TREND" if adx_value > 20 else "RANGING")
        market_is_dead = adx_value < 15
        
        # 8. Signal Logic (V2.6 Enhanced)
        signal = "WAIT"
        unwind_position = False
        reason = []
        quality_flags = []
        score_breakdown = []
        
        # V2.6: VOLATILITY BLOCK (highest priority)
        if volatility_check["block_trades"]:
            signal = "WAIT"
            reason.append(f"VOLATILITY BLOCK: ATR {volatility_check['atr_ratio']:.1f}x average ({volatility_check['regime']})")
            quality_flags.append("VOLATILITY_BLOCK")
        
        # V2.6: DEAD MARKET BLOCK
        elif market_is_dead:
            signal = "WAIT"
            reason.append(f"DEAD MARKET: ADX {adx_value:.1f} (no trend)")
            quality_flags.append("DEAD_MARKET")
        
        else:
            # EXIT CHECKS (V2.6: Use Bear Trap Protection)
            if bear_trap_check["confirmed_breakdown"]:
                unwind_position = True
                reason.append(f"Confirmed Breakdown: 2 candles below EMA200*{100-bear_trap_check['buffer_pct']:.0f}%")
            elif bear_trap_check["warning_breakdown"]:
                # Single candle below - just warn, don't sell yet
                quality_flags.append("BREAKDOWN_WARNING")
                reason.append(f"Breakdown Warning: 1 candle below threshold, watching...")
            elif rsi_value > rsi_exit_threshold:
                unwind_position = True
                reason.append(f"RSI Overextended (>{rsi_exit_threshold})")
            
            if unwind_position:
                signal = "SELL"
            else:
                # ENTRY EVALUATION
                trend_ok = bullish_regime and btc_bullish
                rsi_ok = rsi_value < dynamic_rsi_threshold
                
                volume_ok = volume_analysis["type"] != "distribution"
                volume_bonus = volume_analysis["type"] == "accumulation"
                
                momentum_ok = macd_bullish or macd_accelerating
                
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
                        reason.append(f"Trend: {trend_strength}, ADX: {adx_regime}")
                        
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
                        if adx_value > 30:
                            quality_flags.append("STRONG_ADX")
                        if momentum_override["override_active"]:
                            quality_flags.append("MOMENTUM_OVERRIDE")
                        
                        # V2.6: Funding Rate Warning
                        if funding_rate.get("extreme"):
                            quality_flags.append("EXTREME_FUNDING")
                            reason.append(f"⚠️ Funding Rate extreme: {funding_rate['rate']:.3f}%")
                        elif funding_rate.get("warning"):
                            quality_flags.append("HIGH_FUNDING")
                else:
                    if not bullish_regime:
                        reason.append("Price below trend (EMA filter)")
                    if not btc_bullish:
                        reason.append("BTC bearish (correlation filter)")
        
        # 9. Score Breakdown Calculation (V2.6 Enhanced)
        trend_score = 40 if trend_strength == "STRONG_BULL" else (20 if trend_strength == "WEAK_BULL" else 0)
        rsi_score = 25 if (25 <= rsi_value <= 35) else (15 if (20 <= rsi_value <= 45) else (10 if rsi_value < 20 else 0))
        volume_score = 20 if volume_analysis["type"] == "accumulation" else (-25 if volume_analysis["type"] == "distribution" else 0)
        macd_score = 15 if macd_crossover else (10 if (macd_accelerating and macd_bullish) else (5 if macd_bullish else 0))
        btc_penalty = -15 if not btc_bullish else 0
        mtf_bonus = 10 if mtf_analysis["alignment_status"] == "FULL_ALIGNMENT" else (5 if mtf_analysis["alignment_status"] == "PARTIAL_ALIGNMENT" else 0)
        
        # V2.6: ADX bonus
        adx_bonus = 10 if adx_value > 30 else (5 if adx_value > 25 else 0)
        
        # V2.6: Volatility penalty
        volatility_penalty = -15 if volatility_check["is_chaotic"] else (-5 if volatility_check["is_elevated"] else 0)
        
        raw_score = trend_score + rsi_score + volume_score + macd_score + btc_penalty + mtf_bonus + adx_bonus + volatility_penalty
        
        score_breakdown = [
            f"Trend: {'+' if trend_score >= 0 else ''}{trend_score} ({trend_strength})",
            f"RSI: {'+' if rsi_score >= 0 else ''}{rsi_score} ({rsi_value:.1f})",
            f"Volume: {'+' if volume_score >= 0 else ''}{volume_score} ({volume_analysis['type']})",
            f"MACD: {'+' if macd_score >= 0 else ''}{macd_score}",
            f"MTF: {'+' if mtf_bonus >= 0 else ''}{mtf_bonus} ({mtf_analysis['alignment_status']})",
            f"ADX: {'+' if adx_bonus >= 0 else ''}{adx_bonus} ({adx_value:.1f})",
        ]
        if btc_penalty != 0:
            score_breakdown.append(f"BTC: {btc_penalty}")
        if volatility_penalty != 0:
            score_breakdown.append(f"Volatility: {volatility_penalty}")
        
        score_breakdown_str = " | ".join(score_breakdown)
        
        # 10. ATR-Based Risk/Reward
        if signal == "SELL" or signal == "WAIT":
            suggested_stop_loss = 0
            suggested_target = 0
        else:
            atr_stop_distance = 2 * atr_value
            suggested_stop_loss = current_price - atr_stop_distance
            risk = current_price - suggested_stop_loss
            suggested_target = current_price + (risk * 2)
        
        # 11. Visuals
        buf = io.BytesIO()
        mc = mpf.make_marketcolors(up='#2ebd85', down='#f6465d', volume='in')
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', rc={'font.size': 12})
        
        add_plots = [
            mpf.make_addplot(df['EMA_50'].tail(80), color='orange', width=1.5),
            mpf.make_addplot(df['EMA_200'].tail(80), color='blue', width=2.0),
            mpf.make_addplot(df['MACD_Hist'].tail(80), type='bar', panel=2, color='dimgray', secondary_y=False)
        ]
        mpf.plot(df.tail(80), type='candle', volume=True, addplot=add_plots, style=s, 
                 figsize=(10, 10), panel_ratios=(4, 1, 1), savefig=dict(fname=buf, format='png'))
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # 12. Output JSON (V2.6 Enhanced)
        result = {
            "coin": coin,
            "price": round(current_price, 4),
            "risk_profile": risk_profile,
            "indicators": {
                "rsi": round(rsi_value, 2),
                "ema_50": round(ema_50, 4),
                "ema_200": round(ema_200, 4),
                "atr": round(atr_value, 6),
                "adx": round(adx_value, 2),
                "macd_histogram": round(macd_hist_current, 6),
                "macd_accelerating": macd_accelerating
            },
            "market_context": {
                "trend_strength": trend_strength,
                "btc_bullish": btc_bullish,
                "rsi_exit_threshold": rsi_exit_threshold,
                "ema_spread_pct": round(ema_spread * 100, 2),
                "adx_regime": adx_regime
            },
            "volume_analysis": volume_analysis,
            "momentum": {
                "macd_bullish": macd_bullish,
                "macd_accelerating": macd_accelerating,
                "macd_crossover": macd_crossover
            },
            "multi_timeframe": mtf_analysis,
            # V2.6 New Analysis
            "volatility": volatility_check,
            "bear_trap_protection": bear_trap_check,
            "momentum_override": momentum_override,
            "funding_rate": funding_rate,
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
                "score_breakdown": score_breakdown_str,
                "raw_score": max(0, min(100, raw_score))
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
