import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import csv
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from PIL import Image
import json

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================
# CONFIGURATION
# ========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# ========================
# STOCKS + INDICES (62 symbols)
# ========================
STOCKS_INDICES = {
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "TCS": {"symbol": "TCS", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ"},
    "ITC": {"symbol": "ITC", "segment": "NSE_EQ"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ"},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
    "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ"},
    "LT": {"symbol": "LT", "segment": "NSE_EQ"},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ"},
    "ASIANPAINT": {"symbol": "ASIANPAINT", "segment": "NSE_EQ"},
    "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ"},
    "HCLTECH": {"symbol": "HCLTECH", "segment": "NSE_EQ"},
    "SUNPHARMA": {"symbol": "SUNPHARMA", "segment": "NSE_EQ"},
    "TITAN": {"symbol": "TITAN", "segment": "NSE_EQ"},
    "ULTRACEMCO": {"symbol": "ULTRACEMCO", "segment": "NSE_EQ"},
    "NESTLEIND": {"symbol": "NESTLEIND", "segment": "NSE_EQ"},
    "BAJAJFINSV": {"symbol": "BAJAJFINSV", "segment": "NSE_EQ"},
    "WIPRO": {"symbol": "WIPRO", "segment": "NSE_EQ"},
    "ADANIENT": {"symbol": "ADANIENT", "segment": "NSE_EQ"},
    "ONGC": {"symbol": "ONGC", "segment": "NSE_EQ"},
    "NTPC": {"symbol": "NTPC", "segment": "NSE_EQ"},
    "POWERGRID": {"symbol": "POWERGRID", "segment": "NSE_EQ"},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ"},
    "M&M": {"symbol": "M&M", "segment": "NSE_EQ"},
    "TECHM": {"symbol": "TECHM", "segment": "NSE_EQ"},
    "TATASTEEL": {"symbol": "TATASTEEL", "segment": "NSE_EQ"},
    "INDUSINDBK": {"symbol": "INDUSINDBK", "segment": "NSE_EQ"},
    "JSWSTEEL": {"symbol": "JSWSTEEL", "segment": "NSE_EQ"},
    "DRREDDY": {"symbol": "DRREDDY", "segment": "NSE_EQ"},
    "CIPLA": {"symbol": "CIPLA", "segment": "NSE_EQ"},
    "EICHERMOT": {"symbol": "EICHERMOT", "segment": "NSE_EQ"},
    "APOLLOHOSP": {"symbol": "APOLLOHOSP", "segment": "NSE_EQ"},
    "HINDALCO": {"symbol": "HINDALCO", "segment": "NSE_EQ"},
    "COALINDIA": {"symbol": "COALINDIA", "segment": "NSE_EQ"},
    "ADANIPORTS": {"symbol": "ADANIPORTS", "segment": "NSE_EQ"},
    "BRITANNIA": {"symbol": "BRITANNIA", "segment": "NSE_EQ"},
    "DIVISLAB": {"symbol": "DIVISLAB", "segment": "NSE_EQ"},
    "BPCL": {"symbol": "BPCL", "segment": "NSE_EQ"},
    "GRASIM": {"symbol": "GRASIM", "segment": "NSE_EQ"},
    "HEROMOTOCO": {"symbol": "HEROMOTOCO", "segment": "NSE_EQ"},
    "SHRIRAMFIN": {"symbol": "SHRIRAMFIN", "segment": "NSE_EQ"},
    "TRENT": {"symbol": "TRENT", "segment": "NSE_EQ"},
    "BAJAJ-AUTO": {"symbol": "BAJAJ-AUTO", "segment": "NSE_EQ"},
    "LTIM": {"symbol": "LTIM", "segment": "NSE_EQ"},
    "SBILIFE": {"symbol": "SBILIFE", "segment": "NSE_EQ"},
    "HDFCLIFE": {"symbol": "HDFCLIFE", "segment": "NSE_EQ"},
    "ZOMATO": {"symbol": "ZOMATO", "segment": "NSE_EQ"},
    "PIDILITIND": {"symbol": "PIDILITIND", "segment": "NSE_EQ"},
    "DMART": {"symbol": "DMART", "segment": "NSE_EQ"},
    "ADANIGREEN": {"symbol": "ADANIGREEN", "segment": "NSE_EQ"},
    "IRCTC": {"symbol": "IRCTC", "segment": "NSE_EQ"},
    "PAYTM": {"symbol": "PAYTM", "segment": "NSE_EQ"},
    "NYKAA": {"symbol": "NYKAA", "segment": "NSE_EQ"},
    "POLICYBZR": {"symbol": "POLICYBZR", "segment": "NSE_EQ"},
    "GODREJCP": {"symbol": "GODREJCP", "segment": "NSE_EQ"},
    "SIEMENS": {"symbol": "SIEMENS", "segment": "NSE_EQ"},
}


# ========================
# MOMENTUM SCANNER
# ========================
class MomentumScanner:
    """
    Pure Python momentum & breakout detection
    No complex indicators - just price action!
    """
    
    @staticmethod
    def scan_momentum(candles, spot_price):
        """
        Main scanner - detect momentum, breakouts, breakdowns
        Returns: signal dict with setup details
        """
        if not candles or len(candles) < 50:
            return None
        
        try:
            signal = {
                'type': None,  # BULLISH_MOMENTUM, BEARISH_MOMENTUM, BREAKOUT, BREAKDOWN
                'strength': 0,  # 0-100
                'entry': None,
                'target1': None,
                'target2': None,
                'sl': None,
                'risk_reward': 0,
                'confidence': 0,
                'reasons': []
            }
            
            # Extract price data
            closes = [c['close'] for c in candles]
            highs = [c['high'] for c in candles]
            lows = [c['low'] for c in candles]
            volumes = [c['volume'] for c in candles]
            
            # Current price
            current = closes[-1]
            
            # 1. SWING HIGHS/LOWS (Last 50 candles)
            swing_high = max(highs[-50:])
            swing_low = min(lows[-50:])
            
            # Recent range
            recent_high = max(highs[-20:])
            recent_low = min(lows[-20:])
            
            # 2. PRICE MOMENTUM
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50
            
            # Short term trend
            short_trend = "BULLISH" if sma_20 > sma_50 else "BEARISH"
            
            # Price vs moving averages
            above_sma20 = current > sma_20
            above_sma50 = current > sma_50
            
            # 3. VOLUME CONFIRMATION
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_spike = current_volume > (avg_volume * 1.5)
            
            # 4. CONSECUTIVE CANDLES (Momentum)
            green_candles = 0
            red_candles = 0
            
            for i in range(-5, 0):
                if closes[i] > closes[i-1]:
                    green_candles += 1
                else:
                    red_candles += 1
            
            # 5. RANGE BREAKOUT/BREAKDOWN DETECTION
            range_pct = ((swing_high - swing_low) / swing_low) * 100
            
            # Near resistance
            near_resistance = (swing_high - current) / swing_high < 0.02
            
            # Near support
            near_support = (current - swing_low) / swing_low < 0.02
            
            # 6. PRICE CHANGE (Last 10 candles)
            price_change_10 = ((current - closes[-10]) / closes[-10]) * 100
            
            # ==========================================
            # SIGNAL DETECTION LOGIC
            # ==========================================
            
            # A. BULLISH MOMENTUM
            if (short_trend == "BULLISH" and 
                above_sma20 and above_sma50 and
                green_candles >= 3 and
                price_change_10 > 1.5 and
                volume_spike):
                
                signal['type'] = "BULLISH_MOMENTUM"
                signal['strength'] = min(100, int(price_change_10 * 10))
                signal['entry'] = current
                signal['target1'] = current + (current * 0.02)  # 2%
                signal['target2'] = current + (current * 0.035)  # 3.5%
                signal['sl'] = min(recent_low, sma_20 * 0.98)
                
                signal['reasons'].append(f"Strong bullish momentum - {green_candles} consecutive green candles")
                signal['reasons'].append(f"Price gained {price_change_10:.1f}% in last 10 candles")
                signal['reasons'].append(f"Volume spike: {current_volume/avg_volume:.1f}x average")
                signal['reasons'].append(f"Trading above both SMA20 (₹{sma_20:.1f}) and SMA50 (₹{sma_50:.1f})")
                
                signal['confidence'] = 75
            
            # B. BEARISH MOMENTUM
            elif (short_trend == "BEARISH" and 
                  not above_sma20 and not above_sma50 and
                  red_candles >= 3 and
                  price_change_10 < -1.5 and
                  volume_spike):
                
                signal['type'] = "BEARISH_MOMENTUM"
                signal['strength'] = min(100, int(abs(price_change_10) * 10))
                signal['entry'] = current
                signal['target1'] = current - (current * 0.02)  # 2%
                signal['target2'] = current - (current * 0.035)  # 3.5%
                signal['sl'] = max(recent_high, sma_20 * 1.02)
                
                signal['reasons'].append(f"Strong bearish momentum - {red_candles} consecutive red candles")
                signal['reasons'].append(f"Price dropped {abs(price_change_10):.1f}% in last 10 candles")
                signal['reasons'].append(f"Volume spike: {current_volume/avg_volume:.1f}x average")
                signal['reasons'].append(f"Trading below both SMA20 (₹{sma_20:.1f}) and SMA50 (₹{sma_50:.1f})")
                
                signal['confidence'] = 75
            
            # C. BREAKOUT (Above resistance)
            elif (near_resistance and 
                  current > recent_high and
                  volume_spike and
                  green_candles >= 2 and
                  short_trend == "BULLISH"):
                
                signal['type'] = "BREAKOUT"
                signal['strength'] = 85
                signal['entry'] = current
                signal['target1'] = swing_high + (swing_high * 0.02)
                signal['target2'] = swing_high + (swing_high * 0.04)
                signal['sl'] = recent_high * 0.98
                
                signal['reasons'].append(f"Breakout above resistance at ₹{swing_high:.1f}")
                signal['reasons'].append(f"Strong volume confirmation: {current_volume/avg_volume:.1f}x")
                signal['reasons'].append(f"Previous range: ₹{swing_low:.1f} - ₹{swing_high:.1f} ({range_pct:.1f}%)")
                signal['reasons'].append(f"{green_candles} consecutive bullish candles")
                
                signal['confidence'] = 80
            
            # D. BREAKDOWN (Below support)
            elif (near_support and
                  current < recent_low and
                  volume_spike and
                  red_candles >= 2 and
                  short_trend == "BEARISH"):
                
                signal['type'] = "BREAKDOWN"
                signal['strength'] = 85
                signal['entry'] = current
                signal['target1'] = swing_low - (swing_low * 0.02)
                signal['target2'] = swing_low - (swing_low * 0.04)
                signal['sl'] = recent_low * 1.02
                
                signal['reasons'].append(f"Breakdown below support at ₹{swing_low:.1f}")
                signal['reasons'].append(f"Strong volume confirmation: {current_volume/avg_volume:.1f}x")
                signal['reasons'].append(f"Previous range: ₹{swing_low:.1f} - ₹{swing_high:.1f} ({range_pct:.1f}%)")
                signal['reasons'].append(f"{red_candles} consecutive bearish candles")
                
                signal['confidence'] = 80
            
            # E. SIDEWAYS - NO CLEAR MOMENTUM
            else:
                return None
            
            # Calculate risk:reward
            if signal['sl'] and signal['target1']:
                risk = abs(signal['entry'] - signal['sl'])
                reward = abs(signal['target1'] - signal['entry'])
                signal['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
            
            # Minimum confidence filter
            if signal['confidence'] < 70:
                return None
            
            # Minimum risk:reward filter
            if signal['risk_reward'] < 1.5:
                return None
            
            return signal
            
        except Exception as e:
            logger.error(f"Momentum scan error: {e}")
            return None
    
    @staticmethod
    def detect_candlestick_patterns(candles):
        """Detect key candlestick patterns"""
        patterns = []
        
        if len(candles) < 3:
            return patterns
        
        last = candles[-1]
        prev = candles[-2]
        
        # Calculate components
        body_last = abs(last['close'] - last['open'])
        range_last = last['high'] - last['low']
        
        body_prev = abs(prev['close'] - prev['open'])
        
        # 1. DOJI
        if range_last > 0 and body_last < (range_last * 0.1):
            patterns.append("🔵 DOJI - Indecision")
        
        # 2. BULLISH HAMMER
        if last['close'] > last['open']:
            lower_wick = last['open'] - last['low']
            upper_wick = last['high'] - last['close']
            if lower_wick > (body_last * 2) and upper_wick < (body_last * 0.5):
                patterns.append("🔨 HAMMER - Bullish Reversal")
        
        # 3. SHOOTING STAR
        if last['close'] < last['open']:
            upper_wick = last['high'] - last['open']
            lower_wick = last['close'] - last['low']
            if upper_wick > (body_last * 2) and lower_wick < (body_last * 0.5):
                patterns.append("⭐ SHOOTING STAR - Bearish Reversal")
        
        # 4. BULLISH ENGULFING
        if prev['close'] < prev['open'] and last['close'] > last['open']:
            if last['open'] <= prev['close'] and last['close'] >= prev['open']:
                patterns.append("🟢 BULLISH ENGULFING - Strong Buy")
        
        # 5. BEARISH ENGULFING
        if prev['close'] > prev['open'] and last['close'] < last['open']:
            if last['open'] >= prev['close'] and last['close'] <= prev['open']:
                patterns.append("🔴 BEARISH ENGULFING - Strong Sell")
        
        # 6. MORNING STAR (3-candle pattern)
        if len(candles) >= 3:
            c1 = candles[-3]
            c2 = candles[-2]
            c3 = candles[-1]
            
            # First bearish, second small, third bullish
            if (c1['close'] < c1['open'] and 
                abs(c2['close'] - c2['open']) < body_last * 0.5 and
                c3['close'] > c3['open'] and
                c3['close'] > (c1['open'] + c1['close']) / 2):
                patterns.append("🌅 MORNING STAR - Bullish Reversal")
        
        # 7. EVENING STAR
        if len(candles) >= 3:
            c1 = candles[-3]
            c2 = candles[-2]
            c3 = candles[-1]
            
            # First bullish, second small, third bearish
            if (c1['close'] > c1['open'] and
                abs(c2['close'] - c2['open']) < body_last * 0.5 and
                c3['close'] < c3['open'] and
                c3['close'] < (c1['open'] + c1['close']) / 2):
                patterns.append("🌆 EVENING STAR - Bearish Reversal")
        
        return patterns


# ========================
# OPTION CHAIN ANALYZER
# ========================
class OptionChainAnalyzer:
    """Analyze option chain for PCR, OI, Max Pain"""
    
    @staticmethod
    def analyze(oc_data, spot_price):
        """
        Option chain comprehensive analysis
        Returns: dict with PCR, sentiment, max pain, key strikes
        """
        try:
            if not oc_data or 'oc' not in oc_data:
                return None
            
            oc = oc_data.get('oc', {})
            strikes = sorted([float(s) for s in oc.keys()])
            
            if not strikes:
                return None
            
            # Find ATM strike
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            # Get ATM data
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            ce = atm_data.get('ce', {})
            pe = atm_data.get('pe', {})
            
            ce_oi = ce.get('oi', 0)
            pe_oi = pe.get('oi', 0)
            
            # PCR calculation
            pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
            
            # Sentiment
            if pcr > 1.3:
                sentiment = "STRONG BULLISH 🟢🟢"
            elif pcr > 1.1:
                sentiment = "BULLISH 🟢"
            elif pcr < 0.7:
                sentiment = "STRONG BEARISH 🔴🔴"
            elif pcr < 0.9:
                sentiment = "BEARISH 🔴"
            else:
                sentiment = "NEUTRAL 🟡"
            
            # Find highest OI strikes (resistance/support)
            ce_oi_list = []
            pe_oi_list = []
            
            for strike_str in oc.keys():
                strike = float(strike_str)
                data = oc[strike_str]
                
                if 'ce' in data:
                    ce_oi_list.append((strike, data['ce'].get('oi', 0)))
                if 'pe' in data:
                    pe_oi_list.append((strike, data['pe'].get('oi', 0)))
            
            # Sort by OI
            ce_oi_list.sort(key=lambda x: x[1], reverse=True)
            pe_oi_list.sort(key=lambda x: x[1], reverse=True)
            
            # Max OI strikes
            max_ce_oi_strike = ce_oi_list[0][0] if ce_oi_list else None
            max_pe_oi_strike = pe_oi_list[0][0] if pe_oi_list else None
            
            # Calculate Max Pain
            max_pain = OptionChainAnalyzer._calculate_max_pain(oc)
            
            # IV analysis
            atm_ce_iv = ce.get('implied_volatility', 0)
            atm_pe_iv = pe.get('implied_volatility', 0)
            avg_iv = (atm_ce_iv + atm_pe_iv) / 2 if (atm_ce_iv and atm_pe_iv) else 0
            
            # Build result
            result = {
                'atm_strike': atm_strike,
                'spot': spot_price,
                'pcr': pcr,
                'sentiment': sentiment,
                'max_pain': max_pain,
                'resistance_strike': max_ce_oi_strike,
                'support_strike': max_pe_oi_strike,
                'atm_ce_oi': ce_oi,
                'atm_pe_oi': pe_oi,
                'atm_ce_ltp': ce.get('last_price', 0),
                'atm_pe_ltp': pe.get('last_price', 0),
                'avg_iv': round(avg_iv, 1)
            }
            
            # Additional insights
            result['insights'] = []
            
            if pcr > 1.2:
                result['insights'].append("High PCR suggests strong call writing - bullish")
            elif pcr < 0.8:
                result['insights'].append("Low PCR suggests strong put writing - bearish")
            
            if max_pain:
                if spot_price > max_pain:
                    result['insights'].append(f"Price above Max Pain - possible pullback to ₹{max_pain:.0f}")
                elif spot_price < max_pain:
                    result['insights'].append(f"Price below Max Pain - possible rally to ₹{max_pain:.0f}")
            
            if max_ce_oi_strike and spot_price > (max_ce_oi_strike * 0.98):
                result['insights'].append(f"Near resistance at ₹{max_ce_oi_strike:.0f} (High CE OI)")
            
            if max_pe_oi_strike and spot_price < (max_pe_oi_strike * 1.02):
                result['insights'].append(f"Near support at ₹{max_pe_oi_strike:.0f} (High PE OI)")
            
            return result
            
        except Exception as e:
            logger.error(f"Option chain analysis error: {e}")
            return None
    
    @staticmethod
    def _calculate_max_pain(oc):
        """Calculate max pain strike"""
        try:
            pain_values = {}
            
            strikes = [float(s) for s in oc.keys()]
            
            for test_strike in strikes:
                total_pain = 0
                
                for strike_str in oc.keys():
                    strike = float(strike_str)
                    data = oc[strike_str]
                    
                    ce_oi = data.get('ce', {}).get('oi', 0)
                    pe_oi = data.get('pe', {}).get('oi', 0)
                    
                    # CE pain
                    if test_strike > strike:
                        total_pain += (test_strike - strike) * ce_oi
                    
                    # PE pain
                    if test_strike < strike:
                        total_pain += (strike - test_strike) * pe_oi
                
                pain_values[test_strike] = total_pain
            
            # Find strike with minimum pain
            if pain_values:
                max_pain_strike = min(pain_values, key=pain_values.get)
                return max_pain_strike
            
            return None
            
        except:
            return None


# ========================
# CHART GENERATOR
# ========================
class ChartGenerator:
    @staticmethod
    def create_annotated_chart(candles, symbol, spot_price, signal, oc_analysis):
        """
        Create white background chart with all annotations
        """
        try:
            # Prepare DataFrame
            df_data = []
            for candle in candles[-100:]:  # Last 100 candles
                timestamp = candle.get('timestamp', '')
                df_data.append({
                    'Date': pd.to_datetime(timestamp) if timestamp else pd.Timestamp.now(),
                    'Open': float(candle['open']),
                    'High': float(candle['high']),
                    'Low': float(candle['low']),
                    'Close': float(candle['close']),
                    'Volume': int(float(candle['volume']))
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            
            if len(df) < 2:
                return None
            
            # Style
            mc = mpf.make_marketcolors(
                up='#00cc66',
                down='#ff3366',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle=':',
                gridcolor='#e0e0e0',
                facecolor='white',
                figcolor='white',
                y_on_right=False
            )
            
            # Create plot
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f'\n{symbol} | ₹{spot_price:,.2f} | {signal["type"]}',
                ylabel='Price (₹)',
                ylabel_lower='Volume',
                figsize=(16, 10),
                returnfig=True,
                tight_layout=True
            )
            
            ax_price = axes[0]
            
            # Add trade levels
            if signal['entry']:
                ax_price.axhline(y=signal['entry'], color='blue', linestyle='--', linewidth=2)
                ax_price.text(len(df)*0.02, signal['entry'], f"  ENTRY: ₹{signal['entry']:,.1f}", 
                            color='blue', fontsize=11, fontweight='bold', va='center')
            
            if signal['target1']:
                ax_price.axhline(y=signal['target1'], color='green', linestyle='--', linewidth=2)
                ax_price.text(len(df)*0.02, signal['target1'], f"  T1: ₹{signal['target1']:,.1f}", 
                            color='green', fontsize=10, fontweight='bold', va='center')
            
            if signal['target2']:
                ax_price.axhline(y=signal['target2'], color='darkgreen', linestyle='--', linewidth=2)
                ax_price.text(len(df)*0.02, signal['target2'], f"  T2: ₹{signal['target2']:,.1f}", 
                            color='darkgreen', fontsize=10, fontweight='bold', va='center')
            
            if signal['sl']:
                ax_price.axhline(y=signal['sl'], color='red', linestyle='--', linewidth=2.5)
                ax_price.text(len(df)*0.02, signal['sl'], f"  STOP LOSS: ₹{signal['sl']:,.1f}", 
                            color='red', fontsize=10, fontweight='bold', va='center')
            
            # Option chain levels
            if oc_analysis:
                if oc_analysis['resistance_strike']:
                    ax_price.axhline(y=oc_analysis['resistance_strike'], color='purple', 
                                   linestyle=':', linewidth=1.5, alpha=0.7)
                    ax_price.text(len(df)*0.98, oc_analysis['resistance_strike'], 
                                f"Resistance: ₹{oc_analysis['resistance_strike']:,.0f}  ", 
                                color='purple', fontsize=9, va='center', ha='right')
                
                if oc_analysis['support_strike']:
                    ax_price.axhline(y=oc_analysis['support_strike'], color='orange', 
                                   linestyle=':', linewidth=1.5, alpha=0.7)
                    ax_price.text(len(df)*0.98, oc_analysis['support_strike'], 
                                f"Support: ₹{oc_analysis['support_strike']:,.0f}  ", 
                                color='orange', fontsize=9, va='center', ha='right')
            
            # Title
            title_text = f'{symbol} | ₹{spot_price:,.2f} | {signal["type"]}'
            if signal['confidence']:
                title_text += f' | Confidence: {signal["confidence"]}%'
            
            ax_price.set_title(title_text, color='black', fontsize=18, fontweight='bold', pad=20)
            
            # Save
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            return None


# ========================
# MAIN BOT
# ========================
class TradingBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.running = True
        self.headers = {
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.security_id_map = {}
        self.momentum_scanner = MomentumScanner()
        self.oc_analyzer = OptionChainAnalyzer()
        self.chart_gen = ChartGenerator()
        
        logger.info("🤖 Trading Bot initialized")
    
    async def load_security_ids(self):
        """Load security IDs from Dhan CSV"""
        try:
            logger.info("Loading security IDs...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code == 200:
                csv_data = response.text.split('\n')
                reader = csv.DictReader(csv_data)
                
                for symbol, info in STOCKS_INDICES.items():
                    segment = info['segment']
                    symbol_name = info['symbol']
                    
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)
                    
                    for row in reader:
                        try:
                            if segment == "IDX_I":
                                if (row.get('SEM_SEGMENT') == 'I' and 
                                    row.get('SEM_TRADING_SYMBOL') == symbol_name):
                                    sec_id = row.get('SEM_SMST_SECURITY_ID')
                                    if sec_id:
                                        self.security_id_map[symbol] = {
                                            'security_id': int(sec_id),
                                            'segment': segment,
                                            'trading_symbol': symbol_name
                                        }
                                        logger.info(f"✅ {symbol}: {sec_id}")
                                        break
                            else:
                                if (row.get('SEM_SEGMENT') == 'E' and 
                                    row.get('SEM_TRADING_SYMBOL') == symbol_name and
                                    row.get('SEM_EXM_EXCH_ID') == 'NSE'):
                                    sec_id = row.get('SEM_SMST_SECURITY_ID')
                                    if sec_id:
                                        self.security_id_map[symbol] = {
                                            'security_id': int(sec_id),
                                            'segment': segment,
                                            'trading_symbol': symbol_name
                                        }
                                        logger.info(f"✅ {symbol}: {sec_id}")
                                        break
                        except:
                            continue
                
                logger.info(f"✅ {len(self.security_id_map)} securities loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading IDs: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Fetch 350+ candles from Dhan"""
        try:
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=10)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exch_seg,
                "instrument": instrument,
                "interval": "5",
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            response = requests.post(
                DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'open' in data:
                    opens = data.get('open', [])
                    highs = data.get('high', [])
                    lows = data.get('low', [])
                    closes = data.get('close', [])
                    volumes = data.get('volume', [])
                    timestamps = data.get('start_Time', [])
                    
                    candles = []
                    for i in range(len(opens)):
                        candles.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else '',
                            'open': opens[i],
                            'high': highs[i],
                            'low': lows[i],
                            'close': closes[i],
                            'volume': volumes[i]
                        })
                    
                    logger.info(f"{symbol}: {len(candles)} candles")
                    return candles
            
            return None
        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """Get nearest expiry"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment
            }
            
            response = requests.post(
                DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data'):
                    return data['data'][0]
            return None
        except:
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Get option chain"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment,
                "Expiry": expiry
            }
            
            response = requests.post(
                DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data']
            return None
        except:
            return None
    
    async def scan_and_alert(self, symbol):
        """Main scanning function"""
        try:
            if symbol not in self.security_id_map:
                return
            
            info = self.security_id_map[symbol]
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Scanning {symbol}...")
            logger.info(f"{'='*60}")
            
            # Get expiry
            expiry = self.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"{symbol}: No expiry")
                return
            
            # Get option chain
            oc_data = self.get_option_chain(security_id, segment, expiry)
            if not oc_data:
                logger.warning(f"{symbol}: No option chain")
                return
            
            spot_price = oc_data.get('last_price', 0)
            
            # Get 350 candles
            candles = self.get_historical_data(security_id, segment, symbol)
            if not candles or len(candles) < 50:
                logger.warning(f"{symbol}: Insufficient candles")
                return
            
            # Momentum scan
            signal = self.momentum_scanner.scan_momentum(candles, spot_price)
            
            if not signal:
                logger.info(f"⏭️ {symbol}: No momentum signal")
                return
            
            logger.info(f"🎯 {symbol}: {signal['type']} detected!")
            logger.info(f"   Strength: {signal['strength']}%")
            logger.info(f"   Confidence: {signal['confidence']}%")
            logger.info(f"   R:R = {signal['risk_reward']}")
            
            # Candlestick patterns
            patterns = self.momentum_scanner.detect_candlestick_patterns(candles)
            
            # Option chain analysis
            oc_analysis = self.oc_analyzer.analyze(oc_data, spot_price)
            
            # Generate chart
            chart_buf = self.chart_gen.create_annotated_chart(
                candles, symbol, spot_price, signal, oc_analysis
            )
            
            if not chart_buf:
                logger.warning(f"Chart generation failed for {symbol}")
                return
            
            # Send to Telegram
            await self.send_alert(symbol, spot_price, signal, patterns, oc_analysis, chart_buf)
            
            logger.info(f"✅ {symbol} alert sent!")
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
    
    async def send_alert(self, symbol, spot_price, signal, patterns, oc_analysis, chart_buf):
        """Send formatted alert to Telegram"""
        try:
            # Chart first
            chart_buf.seek(0)
            caption = f"🎯 *{signal['type']}*\n"
            caption += f"📊 {symbol} | ₹{spot_price:,.2f}\n"
            caption += f"💪 Strength: {signal['strength']}% | Confidence: {signal['confidence']}%"
            
            await self.bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=chart_buf,
                caption=caption,
                parse_mode='Markdown'
            )
            
            # Text alert
            msg = f"🚨 *TRADE ALERT* 🚨\n"
            msg += f"{'='*40}\n\n"
            
            msg += f"📊 *Symbol:* {symbol}\n"
            msg += f"💰 *Spot Price:* ₹{spot_price:,.2f}\n"
            msg += f"🎯 *Signal:* {signal['type']}\n"
            msg += f"💪 *Strength:* {signal['strength']}%\n"
            msg += f"✅ *Confidence:* {signal['confidence']}%\n\n"
            
            msg += f"{'='*40}\n"
            msg += f"📈 *TRADE SETUP*\n"
            msg += f"{'='*40}\n\n"
            
            if signal['entry']:
                msg += f"🎯 *Entry:* ₹{signal['entry']:,.2f}\n"
            
            if signal['target1']:
                gain1 = ((signal['target1'] - signal['entry']) / signal['entry']) * 100
                msg += f"🟢 *Target 1:* ₹{signal['target1']:,.2f} (+{gain1:.1f}%)\n"
            
            if signal['target2']:
                gain2 = ((signal['target2'] - signal['entry']) / signal['entry']) * 100
                msg += f"🟢 *Target 2:* ₹{signal['target2']:,.2f} (+{gain2:.1f}%)\n"
            
            if signal['sl']:
                loss = ((signal['entry'] - signal['sl']) / signal['entry']) * 100
                msg += f"🛑 *Stop Loss:* ₹{signal['sl']:,.2f} (-{loss:.1f}%)\n"
            
            if signal['risk_reward']:
                msg += f"\n📊 *Risk:Reward:* 1:{signal['risk_reward']}\n"
            
            # Reasons
            if signal['reasons']:
                msg += f"\n{'='*40}\n"
                msg += f"💡 *WHY THIS TRADE?*\n"
                msg += f"{'='*40}\n\n"
                for reason in signal['reasons']:
                    msg += f"✓ {reason}\n"
            
            # Patterns
            if patterns:
                msg += f"\n{'='*40}\n"
                msg += f"🕯️ *CANDLESTICK PATTERNS*\n"
                msg += f"{'='*40}\n\n"
                for pattern in patterns:
                    msg += f"{pattern}\n"
            
            # Option chain
            if oc_analysis:
                msg += f"\n{'='*40}\n"
                msg += f"📊 *OPTION CHAIN ANALYSIS*\n"
                msg += f"{'='*40}\n\n"
                
                msg += f"🎯 *ATM Strike:* ₹{oc_analysis['atm_strike']:,.0f}\n"
                msg += f"📈 *PCR Ratio:* {oc_analysis['pcr']}\n"
                msg += f"💭 *Sentiment:* {oc_analysis['sentiment']}\n\n"
                
                msg += f"📞 *Call OI:* {oc_analysis['atm_ce_oi']/1000:.0f}K\n"
                msg += f"📉 *Put OI:* {oc_analysis['atm_pe_oi']/1000:.0f}K\n\n"
                
                if oc_analysis['max_pain']:
                    msg += f"💥 *Max Pain:* ₹{oc_analysis['max_pain']:,.0f}\n"
                
                if oc_analysis['resistance_strike']:
                    msg += f"🔴 *Resistance:* ₹{oc_analysis['resistance_strike']:,.0f}\n"
                
                if oc_analysis['support_strike']:
                    msg += f"🟢 *Support:* ₹{oc_analysis['support_strike']:,.0f}\n"
                
                if oc_analysis['insights']:
                    msg += f"\n💡 *Option Insights:*\n"
                    for insight in oc_analysis['insights']:
                        msg += f"• {insight}\n"
            
            msg += f"\n{'='*40}\n"
            msg += f"⏰ *Time:* {datetime.now().strftime('%d-%m-%Y %H:%M IST')}\n"
            msg += f"{'='*40}"
            
            # Send text
            if len(msg) > 4000:
                parts = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
                for part in parts:
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=part,
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(1)
            else:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg,
                    parse_mode='Markdown'
                )
            
        except Exception as e:
            logger.error(f"Alert sending error: {e}")
    
    async def send_startup_message(self):
        """Startup notification"""
        try:
            msg = "🤖 *MOMENTUM TRADING BOT ACTIVATED!*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            msg += f"📊 *Tracking:* {len(self.security_id_map)} Stocks/Indices\n"
            msg += f"⏱️ *Scan Frequency:* Every 15 minutes\n"
            msg += f"📈 *Timeframe:* 5-minute candles\n"
            msg += f"📊 *Data:* 350+ historical candles\n\n"
            
            msg += "🎯 *DETECTION STRATEGY:*\n"
            msg += "  ✓ Bullish Momentum\n"
            msg += "  ✓ Bearish Momentum\n"
            msg += "  ✓ Breakouts (Above Resistance)\n"
            msg += "  ✓ Breakdowns (Below Support)\n\n"
            
            msg += "📊 *ANALYSIS INCLUDES:*\n"
            msg += "  ✓ Price Action (350+ candles)\n"
            msg += "  ✓ Volume Confirmation\n"
            msg += "  ✓ Swing High/Low Detection\n"
            msg += "  ✓ Candlestick Patterns\n"
            msg += "  ✓ Option Chain (PCR, OI, Max Pain)\n"
            msg += "  ✓ Entry/Target/SL Levels\n"
            msg += "  ✓ Risk:Reward > 1.5\n\n"
            
            msg += "💡 *FILTERS:*\n"
            msg += "  • Minimum Confidence: 70%\n"
            msg += "  • Volume Spike: >1.5x average\n"
            msg += "  • Clear momentum setup\n\n"
            
            msg += "📋 *MONITORING:*\n"
            msg += "  • 2 Indices (NIFTY, BANKNIFTY)\n"
            msg += "  • 60 Top Stocks\n\n"
            
            msg += "🔔 *Status:* ACTIVE ✅\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            
            logger.info("✅ Startup message sent")
        except Exception as e:
            logger.error(f"Startup message error: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("🚀 Starting Momentum Scanner Bot...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load security IDs")
            return
        
        await self.send_startup_message()
        
        all_symbols = list(self.security_id_map.keys())
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 NEW SCAN CYCLE: {timestamp}")
                logger.info(f"{'='*70}\n")
                
                for idx, symbol in enumerate(all_symbols, 1):
                    logger.info(f"📊 [{idx}/{len(all_symbols)}] {symbol}")
                    
                    await self.scan_and_alert(symbol)
                    
                    if idx < len(all_symbols):
                        await asyncio.sleep(8)
                
                logger.info("\n" + "="*70)
                logger.info("✅ SCAN CYCLE COMPLETED!")
                logger.info("⏳ Next scan in 15 minutes...")
                logger.info("="*70 + "\n")
                
                await asyncio.sleep(900)  # 15 minutes
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    try:
        required_vars = {
            'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
            'DHAN_CLIENT_ID': DHAN_CLIENT_ID,
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN
        }
        
        missing = [k for k, v in required_vars.items() if not v]
        
        if missing:
            logger.error("❌ MISSING ENVIRONMENT VARIABLES!")
            logger.error(f"Missing: {', '.join(missing)}")
            exit(1)
        
        logger.info("✅ All environment variables OK")
        logger.info("🚀 Starting bot...")
        
        bot = TradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
