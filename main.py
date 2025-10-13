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
import matplotlib.patches as patches
import mplfinance as mpf
import pandas as pd
from groq import Groq
from openai import OpenAI
from PIL import Image
import time
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# ========================
# 🎯 60 STOCKS + 2 INDICES
# ========================
STOCKS_INDICES = {
    # ==================
    # INDICES (2)
    # ==================
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    
    # ==================
    # NIFTY 50 STOCKS (50)
    # ==================
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
    
    # ==================
    # ADDITIONAL TOP 10 STOCKS (10)
    # ==================
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
# COST TRACKER
# ========================
class CostTracker:
    def __init__(self, budget_per_month=300):
        self.budget = budget_per_month
        self.costs = []
    
    def log_call(self, model, tokens_in, tokens_out):
        """Log API call and calculate cost"""
        if model == "groq":
            cost = 0  # FREE!
        elif model == "gpt-o1-mini":
            # o1-mini pricing
            cost_in = (tokens_in / 1_000_000) * 3  # USD
            cost_out = (tokens_out / 1_000_000) * 12  # USD
            cost = (cost_in + cost_out) * 83  # Convert to INR
        else:
            cost = 0
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_inr": cost
        }
        
        self.costs.append(log_entry)
        return cost
    
    def get_total_cost(self):
        """Get total cost so far"""
        return sum(c['cost_inr'] for c in self.costs)
    
    def print_summary(self):
        """Print cost summary"""
        total = self.get_total_cost()
        remaining = self.budget - total
        
        logger.info("\n" + "="*50)
        logger.info("💰 COST SUMMARY")
        logger.info("="*50)
        logger.info(f"Total spent: ₹{total:.2f}")
        logger.info(f"Budget: ₹{self.budget:.2f}")
        logger.info(f"Remaining: ₹{remaining:.2f}")
        logger.info(f"Usage: {(total/self.budget)*100:.1f}%")
        
        if remaining < 50:
            logger.warning("⚠️ WARNING: Low budget!")
        
        return remaining


# ========================
# GROQ PRE-FILTER ANALYZER
# ========================
class GroqPreFilter:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        logger.info("✅ Groq initialized (FREE - unlimited)")
    
    def quick_scan(self, symbol, candles, spot_price, technical_data, patterns):
        """
        Groq se quick scan - filter karta hai stocks
        Returns: score (0-10) and reason
        """
        try:
            # Last 20 candles ka summary
            recent_candles = candles[-20:] if len(candles) >= 20 else candles
            candle_summary = self._format_candles(recent_candles)
            
            # Technical summary
            tech_summary = self._format_technical(technical_data, spot_price)
            
            # Patterns
            pattern_text = ", ".join(patterns) if patterns else "No patterns"
            
            prompt = f"""
You are a quick filter for Indian stock trading bot.

STOCK: {symbol}
SPOT PRICE: ₹{spot_price:,.2f}

TECHNICAL:
{tech_summary}

PATTERNS: {pattern_text}

RECENT PRICE ACTION (Last 20 candles):
{candle_summary}

YOUR TASK:
Score this setup from 0-10 based on:
1. Trend strength (alignment)
2. Volume confirmation
3. Pattern quality
4. Clear entry/exit levels
5. Risk-reward potential

OUTPUT FORMAT (strictly follow):
SCORE: [0-10]
REASON: [One line explaining score]

Example:
SCORE: 8
REASON: Strong bullish trend, volume spike, breakout above resistance

Now analyze {symbol}:
"""
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150
            )
            
            result = response.choices[0].message.content
            
            # Extract score
            score = 0
            reason = "No clear setup"
            
            lines = result.strip().split('\n')
            for line in lines:
                if 'SCORE:' in line.upper():
                    try:
                        score = int(''.join(filter(str.isdigit, line)))
                    except:
                        score = 0
                elif 'REASON:' in line.upper():
                    reason = line.split(':', 1)[1].strip()
            
            logger.info(f"Groq: {symbol} → Score: {score}/10 | {reason}")
            
            return {
                'score': score,
                'reason': reason,
                'raw_response': result
            }
            
        except Exception as e:
            logger.error(f"Groq error for {symbol}: {e}")
            return {'score': 0, 'reason': 'API error', 'raw_response': ''}
    
    def _format_candles(self, candles):
        """Format last few candles"""
        lines = []
        for i, c in enumerate(candles[-5:], 1):
            change = c['close'] - c['open']
            pct = (change / c['open']) * 100
            candle_type = "GREEN" if change > 0 else "RED"
            lines.append(f"{i}. {candle_type}: O={c['open']:.1f} C={c['close']:.1f} ({pct:+.1f}%)")
        return "\n".join(lines)
    
    def _format_technical(self, tech, price):
        """Format technical data"""
        if not tech:
            return "Technical data unavailable"
        
        return f"""
Current: ₹{price:,.2f}
Trend: {tech.get('trend', 'N/A')}
SMA20: ₹{tech.get('sma_20', 'N/A')} | SMA50: ₹{tech.get('sma_50', 'N/A')}
RSI: {tech.get('rsi', 'N/A')}
Support: ₹{tech.get('support', 'N/A')} | Resistance: ₹{tech.get('resistance', 'N/A')}
Volume: {'HIGH' if tech.get('volume_spike') else 'NORMAL'}
"""


# ========================
# GPT O1-MINI DEEP ANALYZER
# ========================
class GPTo1MiniAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.cost_tracker = CostTracker(budget_per_month=300)
        logger.info("✅ GPT o1-mini initialized")
    
    async def deep_analysis(self, symbol, candles, spot_price, technical_data, 
                           patterns, option_data, groq_reason):
        """
        o1-mini se deep reasoning analysis
        360 candlesticks + option chain combined
        """
        try:
            logger.info(f"🧠 Running o1-mini deep analysis for {symbol}...")
            
            # Last 360 candles (full data for reasoning)
            analysis_candles = candles[-360:] if len(candles) >= 360 else candles
            
            # Format data for AI
            candles_json = self._format_candles_json(analysis_candles)
            option_summary = self._format_option_chain(option_data, spot_price)
            tech_summary = self._format_technical(technical_data, spot_price)
            pattern_text = "\n".join(patterns) if patterns else "No major patterns"
            
            # Comprehensive prompt
            prompt = f"""
You are an EXPERT Indian stock market trader specializing in F&O options trading.

STOCK: {symbol}
CURRENT SPOT PRICE: ₹{spot_price:,.2f}
TIMESTAMP: {datetime.now().strftime('%d-%m-%Y %H:%M IST')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 TECHNICAL INDICATORS:
{tech_summary}

🕯️ CANDLESTICK PATTERNS DETECTED:
{pattern_text}

📈 PRICE ACTION DATA (Last 360 candles - 5min timeframe):
{candles_json}

💹 OPTION CHAIN ANALYSIS:
{option_summary}

🔍 GROQ PRE-FILTER NOTED:
{groq_reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR TASK - DEEP REASONING ANALYSIS:

Think step-by-step and analyze:

1. PRICE ACTION STRUCTURE:
   - Identify swing highs/lows from 360 candles
   - Support/resistance zones
   - Trend structure (higher highs/lows or lower highs/lows)
   - Consolidation vs trending phases
   - Volume profile at key levels

2. MULTI-TIMEFRAME CONTEXT:
   - Short-term trend (last 20 candles)
   - Medium-term trend (last 100 candles)
   - Long-term trend (all 360 candles)
   - Are timeframes aligned?

3. OPTION CHAIN CONFIRMATION:
   - Does PCR support price direction?
   - Are high OI strikes acting as support/resistance?
   - Is IV expanding (breakout) or contracting (range)?
   - Where is max pain vs current price?

4. ENTRY SETUP QUALITY:
   - Is there a clear trigger point?
   - Can we define precise entry level?
   - Is risk manageable with structure-based SL?
   - What's the risk-reward ratio?

5. PROBABILITY ASSESSMENT:
   - What factors support this trade?
   - What could invalidate the setup?
   - Historical behavior at these levels?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FORMAT (Strictly follow):

🎯 TRADE DECISION: [YES/NO/WAIT]

IF YES - Provide complete trade plan:

📊 DIRECTION: [BULLISH/BEARISH]

💰 TRADE SETUP:
Stock/Index: {symbol}
Entry Price: ₹[exact price]
Entry Condition: [Specific trigger - e.g., "Enter on close above ₹2150 with volume > 1.5x avg"]

🎯 TARGETS:
Target 1: ₹[price] ([X]% gain) - Book [50]% position
Target 2: ₹[price] ([X]% gain) - Trail remaining

🛑 STOP LOSS:
SL: ₹[price] ([X]% risk)
Basis: [Structure-based reason - e.g., "Below swing low"]

📊 RISK:REWARD: [X:Y ratio]

⏰ TIMEFRAME: [Intraday/Swing (1-2 days)/Positional (>3 days)]

💡 OPTION STRATEGY (if applicable):
[CE/PE Strike to buy/sell]
Strike: [price]
Reasoning: [Why this strike]

🔥 CONFIDENCE: [X]%

📝 REASONING:
1. [Key factor 1]
2. [Key factor 2]
3. [Key factor 3]
4. [Why risk-reward is favorable]

⚠️ INVALIDATION:
Setup fails if: [Specific condition]

⚠️ RISKS:
- [Risk factor 1]
- [Risk factor 2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IF NO/WAIT - Explain why:

❌ REASON: [Why not trading]
⏳ WATCH FOR: [What conditions would make it tradeable]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
✅ Only recommend if confidence > 70%
✅ Entry must have clear trigger
✅ SL must be structure-based
✅ Risk:Reward must be > 1:1.5
✅ Be conservative - "NO trade" is better than bad trade
✅ Consider option chain confirmation
✅ Account for Indian market volatility

Analyze thoroughly and provide actionable trade plan!
"""
            
            # o1-mini API call
            response = self.client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = response.choices[0].message.content
            
            # Log cost
            usage = response.usage
            cost = self.cost_tracker.log_call(
                "gpt-o1-mini",
                usage.prompt_tokens,
                usage.completion_tokens
            )
            
            logger.info(f"✅ o1-mini analysis done | Cost: ₹{cost:.2f}")
            
            # Check if trade recommended
            trade_decision = "NO"
            if "TRADE DECISION: YES" in analysis.upper():
                trade_decision = "YES"
            elif "TRADE DECISION: WAIT" in analysis.upper():
                trade_decision = "WAIT"
            
            return {
                'decision': trade_decision,
                'analysis': analysis,
                'cost': cost
            }
            
        except Exception as e:
            logger.error(f"❌ o1-mini error for {symbol}: {e}")
            return None
    
    def _format_candles_json(self, candles):
        """Format candles as compact JSON for AI"""
        formatted = []
        for i, c in enumerate(candles, 1):
            formatted.append({
                'n': i,
                'o': round(c['open'], 2),
                'h': round(c['high'], 2),
                'l': round(c['low'], 2),
                'c': round(c['close'], 2),
                'v': int(c['volume'])
            })
        
        # Return as JSON string (compact)
        return json.dumps(formatted, separators=(',', ':'))
    
    def _format_technical(self, tech, price):
        """Format technical indicators"""
        if not tech:
            return "Technical data unavailable"
        
        rsi_status = "Overbought" if tech.get('rsi', 50) > 70 else "Oversold" if tech.get('rsi', 50) < 30 else "Neutral"
        
        return f"""
Current Price: ₹{price:,.2f}
Trend: {tech.get('trend', 'N/A')}
SMA(20): ₹{tech.get('sma_20', 'N/A')}
SMA(50): ₹{tech.get('sma_50', 'N/A')}
RSI(14): {tech.get('rsi', 'N/A')} - {rsi_status}
Support: ₹{tech.get('support', 'N/A')}
Resistance: ₹{tech.get('resistance', 'N/A')}
Volume Status: {'HIGH SPIKE ⚡' if tech.get('volume_spike') else 'Normal'}
Avg Volume: {tech.get('avg_volume', 'N/A')}
"""
    
    def _format_option_chain(self, oc_data, spot):
        """Format option chain data"""
        try:
            if not oc_data or 'oc' not in oc_data:
                return "Option data not available"
            
            oc = oc_data.get('oc', {})
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot))
            
            # ATM data
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            ce = atm_data.get('ce', {})
            pe = atm_data.get('pe', {})
            
            ce_oi = ce.get('oi', 0)
            pe_oi = pe.get('oi', 0)
            pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
            
            sentiment = "BULLISH 🟢" if pcr > 1.2 else "BEARISH 🔴" if pcr < 0.8 else "NEUTRAL 🟡"
            
            # Find highest OI strikes
            ce_oi_strikes = [(float(s), oc[s]['ce']['oi']) for s in oc.keys() if 'ce' in oc[s]]
            pe_oi_strikes = [(float(s), oc[s]['pe']['oi']) for s in oc.keys() if 'pe' in oc[s]]
            
            ce_oi_strikes.sort(key=lambda x: x[1], reverse=True)
            pe_oi_strikes.sort(key=lambda x: x[1], reverse=True)
            
            max_ce_oi_strike = ce_oi_strikes[0][0] if ce_oi_strikes else 0
            max_pe_oi_strike = pe_oi_strikes[0][0] if pe_oi_strikes else 0
            
            return f"""
ATM Strike: ₹{atm_strike:,.0f}
Spot Price: ₹{spot:,.2f}

ATM CALL (CE):
  OI: {ce_oi/1000:.0f}K contracts
  LTP: ₹{ce.get('last_price', 0):.1f}
  IV: {ce.get('implied_volatility', 0):.1f}%

ATM PUT (PE):
  OI: {pe_oi/1000:.0f}K contracts
  LTP: ₹{pe.get('last_price', 0):.1f}
  IV: {pe.get('implied_volatility', 0):.1f}%

PCR RATIO: {pcr} → {sentiment}

MAX OI STRIKES (Key Levels):
  CE: ₹{max_ce_oi_strike:,.0f} (Resistance)
  PE: ₹{max_pe_oi_strike:,.0f} (Support)

Max Pain: ₹{oc_data.get('max_pain', 'N/A')}
"""
        except Exception as e:
            logger.error(f"Option chain format error: {e}")
            return "Option summary error"


# ========================
# TECHNICAL ANALYZER
# ========================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(candles):
        try:
            if not candles or len(candles) < 20:
                return None
            
            closes = [c['close'] for c in candles[-50:]]
            highs = [c['high'] for c in candles[-50:]]
            lows = [c['low'] for c in candles[-50:]]
            volumes = [c['volume'] for c in candles[-50:]]
            
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            
            rsi = TechnicalAnalyzer._calculate_rsi(closes, 14)
            
            resistance = max(highs[-50:]) if len(highs) >= 50 else max(highs)
            support = min(lows[-50:]) if len(lows) >= 50 else min(lows)
            
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            volume_spike = current_volume > (avg_volume * 1.5)
            
            if sma_20 and sma_50:
                trend = "BULLISH" if sma_20 > sma_50 else "BEARISH"
            else:
                trend = "SIDEWAYS"
            
            return {
                'sma_20': round(sma_20, 2) if sma_20 else None,
                'sma_50': round(sma_50, 2) if sma_50 else None,
                'rsi': round(rsi, 2) if rsi else None,
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'trend': trend,
                'volume_spike': volume_spike,
                'avg_volume': int(avg_volume)
            }
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return None
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        try:
            if len(prices) < period + 1:
                return None
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if len(gains) < period:
                return None
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return None
    
    @staticmethod
    def detect_candlestick_patterns(candles):
        patterns = []
        
        if len(candles) < 3:
            return patterns
        
        last = candles[-1]
        prev = candles[-2]
        
        # Doji
        body = abs(last['close'] - last['open'])
        range_size = last['high'] - last['low']
        if body < (range_size * 0.1) and range_size > 0:
            patterns.append("🔵 DOJI (Indecision)")
        
        # Hammer
        if last['close'] > last['open']:
            lower_wick = last['open'] - last['low']
            upper_wick = last['high'] - last['close']
            body = last['close'] - last['open']
            if lower_wick > (body * 2) and upper_wick < body:
                patterns.append("🔨 HAMMER (Bullish Reversal)")
        
        # Shooting Star
        if last['close'] < last['open']:
            upper_wick = last['high'] - last['open']
            lower_wick = last['close'] - last['low']
            body = last['open'] - last['close']
            if upper_wick > (body * 2) and lower_wick < body:
                patterns.append("⭐ SHOOTING STAR (Bearish Reversal)")
        
        # Bullish Engulfing
        if prev['close'] < prev['open'] and last['close'] > last['open']:
            if last['open'] < prev['close'] and last['close'] > prev['open']:
                patterns.append("🟢 BULLISH ENGULFING")
        
        # Bearish Engulfing
        if prev['close'] > prev['open'] and last['close'] < last['open']:
            if last['open'] > prev['close'] and last['close'] < prev['open']:
                patterns.append("🔴 BEARISH ENGULFING")
        
        return patterns


# ========================
# CHART GENERATOR
# ========================
class ChartGenerator:
    @staticmethod
    def create_chart_with_annotations(candles, symbol, spot_price, trade_data):
        """
        White background chart with entry/target/SL annotations
        """
        try:
            # Prepare dataframe
            df_data = []
            for candle in candles:
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
            
            # White background style
            mc = mpf.make_marketcolors(
                up='#00cc66',      # Green candles
                down='#ff3366',    # Red candles
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle=':',
                gridcolor='#cccccc',
                facecolor='white',
                figcolor='white',
                y_on_right=False
            )
            
            # Create figure
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f'\n{symbol} | ₹{spot_price:,.2f} | 5min Chart',
                ylabel='Price (₹)',
                ylabel_lower='Volume',
                figsize=(16, 10),
                returnfig=True,
                tight_layout=True
            )
            
            # Get price axis
            ax_price = axes[0]
            
            # Extract trade levels from trade_data
            entry_price = trade_data.get('entry_price')
            target1_price = trade_data.get('target1_price')
            target2_price = trade_data.get('target2_price')
            sl_price = trade_data.get('sl_price')
            support = trade_data.get('support')
            resistance = trade_data.get('resistance')
            
            # Add horizontal lines
            if entry_price:
                ax_price.axhline(y=entry_price, color='blue', linestyle='--', linewidth=2, label=f'Entry: ₹{entry_price:,.2f}')
                ax_price.text(len(df)*0.02, entry_price, f'  ENTRY: ₹{entry_price:,.2f}', 
                            color='blue', fontsize=10, fontweight='bold', va='center')
            
            if target1_price:
                ax_price.axhline(y=target1_price, color='green', linestyle='--', linewidth=2, label=f'T1: ₹{target1_price:,.2f}')
                ax_price.text(len(df)*0.02, target1_price, f'  TARGET 1: ₹{target1_price:,.2f}', 
                            color='green', fontsize=10, fontweight='bold', va='center')
            
            if target2_price:
                ax_price.axhline(y=target2_price, color='darkgreen', linestyle='--', linewidth=2, label=f'T2: ₹{target2_price:,.2f}')
                ax_price.text(len(df)*0.02, target2_price, f'  TARGET 2: ₹{target2_price:,.2f}', 
                            color='darkgreen', fontsize=10, fontweight='bold', va='center')
            
            if sl_price:
                ax_price.axhline(y=sl_price, color='red', linestyle='--', linewidth=2, label=f'SL: ₹{sl_price:,.2f}')
                ax_price.text(len(df)*0.02, sl_price, f'  STOP LOSS: ₹{sl_price:,.2f}', 
                            color='red', fontsize=10, fontweight='bold', va='center')
            
            if support:
                ax_price.axhline(y=support, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                ax_price.text(len(df)*0.02, support, f'  Support: ₹{support:,.2f}', 
                            color='orange', fontsize=9, va='center')
            
            if resistance:
                ax_price.axhline(y=resistance, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
                ax_price.text(len(df)*0.02, resistance, f'  Resistance: ₹{resistance:,.2f}', 
                            color='purple', fontsize=9, va='center')
            
            # Title styling
            ax_price.set_title(
                f'{symbol} | ₹{spot_price:,.2f} | 5min Chart',
                color='black',
                fontsize=18,
                fontweight='bold',
                pad=20
            )
            
            # Axis styling
            for ax in axes:
                ax.tick_params(colors='black')
                ax.spines['bottom'].set_color('#999')
                ax.spines['top'].set_color('#999')
                ax.spines['left'].set_color('#999')
                ax.spines['right'].set_color('#999')
                ax.xaxis.label.set_color('black')
                ax.yaxis.label.set_color('black')
            
            # Add legend
            ax_price.legend(loc='upper left', fontsize=9)
            
            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            logger.error(f"Chart creation error for {symbol}: {e}")
            return None


# ========================
# MAIN BOT CLASS
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
        self.groq_filter = GroqPreFilter(GROQ_API_KEY)
        self.gpt_analyzer = GPTo1MiniAnalyzer(OPENAI_API_KEY)
        self.tech_analyzer = TechnicalAnalyzer()
        self.chart_gen = ChartGenerator()
        
        logger.info("🤖 Trading Bot initialized")
        logger.info("✅ Groq Pre-Filter: FREE")
        logger.info("✅ GPT o1-mini: ₹1.43 per analysis")
    
    async def load_security_ids(self):
        """Load security IDs from Dhan CSV"""
        try:
            logger.info("Loading security IDs from Dhan...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code == 200:
                csv_data = response.text.split('\n')
                reader = csv.DictReader(csv_data)
                
                for symbol, info in STOCKS_INDICES.items():
                    segment = info['segment']
                    symbol_name = info['symbol']
                    
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
                    
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)
                
                logger.info(f"✅ {len(self.security_id_map)} securities loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading IDs: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Fetch 360+ candles (5min) from Dhan"""
        try:
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=10)  # 10 days for 360+ candles
            
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
                    
                    logger.info(f"{symbol}: {len(candles)} candles fetched")
                    return candles
            
            return None
        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """Get nearest expiry from Dhan"""
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
        except Exception as e:
            logger.error(f"Expiry error: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Get option chain from Dhan"""
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
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None
    
    def extract_trade_levels(self, analysis_text):
        """Extract entry/target/SL from o1-mini analysis"""
        try:
            trade_data = {}
            
            lines = analysis_text.split('\n')
            for line in lines:
                line_upper = line.upper()
                
                # Entry price
                if 'ENTRY PRICE:' in line_upper:
                    try:
                        price = float(''.join(filter(lambda x: x.isdigit() or x == '.', line.split('₹')[1].split()[0])))
                        trade_data['entry_price'] = price
                    except:
                        pass
                
                # Target 1
                if 'TARGET 1:' in line_upper and '₹' in line:
                    try:
                        price = float(''.join(filter(lambda x: x.isdigit() or x == '.', line.split('₹')[1].split()[0])))
                        trade_data['target1_price'] = price
                    except:
                        pass
                
                # Target 2
                if 'TARGET 2:' in line_upper and '₹' in line:
                    try:
                        price = float(''.join(filter(lambda x: x.isdigit() or x == '.', line.split('₹')[1].split()[0])))
                        trade_data['target2_price'] = price
                    except:
                        pass
                
                # Stop Loss
                if 'SL:' in line_upper and '₹' in line:
                    try:
                        price = float(''.join(filter(lambda x: x.isdigit() or x == '.', line.split('₹')[1].split()[0])))
                        trade_data['sl_price'] = price
                    except:
                        pass
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Error extracting trade levels: {e}")
            return {}
    
    async def analyze_and_send(self, symbol):
        """Complete analysis pipeline: Groq filter → o1-mini deep → Telegram alert"""
        try:
            if symbol not in self.security_id_map:
                logger.warning(f"⚠️ {symbol} - No security ID")
                return
            
            info = self.security_id_map[symbol]
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Analyzing {symbol}...")
            logger.info(f"{'='*60}")
            
            # 1. Get expiry
            expiry = self.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"{symbol}: No expiry available")
                return
            
            # 2. Get option chain
            oc_data = self.get_option_chain(security_id, segment, expiry)
            if not oc_data:
                logger.warning(f"{symbol}: No option chain")
                return
            
            spot_price = oc_data.get('last_price', 0)
            
            # 3. Get 360+ candles
            candles = self.get_historical_data(security_id, segment, symbol)
            if not candles or len(candles) < 100:
                logger.warning(f"{symbol}: Insufficient candles ({len(candles) if candles else 0})")
                return
            
            logger.info(f"✅ {symbol}: {len(candles)} candles fetched")
            
            # 4. Technical analysis
            technical_data = self.tech_analyzer.calculate_indicators(candles)
            
            # 5. Pattern detection
            patterns = self.tech_analyzer.detect_candlestick_patterns(candles)
            
            # 6. 🚀 GROQ PRE-FILTER (FREE)
            logger.info(f"🔍 Running Groq pre-filter for {symbol}...")
            groq_result = self.groq_filter.quick_scan(
                symbol, candles, spot_price, technical_data, patterns
            )
            
            groq_score = groq_result['score']
            groq_reason = groq_result['reason']
            
            # If Groq score < 6, skip deep analysis
            if groq_score < 6:
                logger.info(f"⏭️ {symbol} skipped (Groq score: {groq_score}/10) - {groq_reason}")
                return
            
            logger.info(f"✅ {symbol} passed Groq filter (Score: {groq_score}/10)")
            
            # 7. 🧠 GPT O1-MINI DEEP ANALYSIS (PAID)
            logger.info(f"🧠 Running o1-mini deep analysis for {symbol}...")
            
            gpt_result = await self.gpt_analyzer.deep_analysis(
                symbol, candles, spot_price, technical_data, 
                patterns, oc_data, groq_reason
            )
            
            if not gpt_result:
                logger.warning(f"⚠️ o1-mini analysis failed for {symbol}")
                return
            
            decision = gpt_result['decision']
            analysis = gpt_result['analysis']
            cost = gpt_result['cost']
            
            logger.info(f"✅ o1-mini decision: {decision} | Cost: ₹{cost:.2f}")
            
            # 8. Only send if trade recommended
            if decision != "YES":
                logger.info(f"⏸️ {symbol} - No trade signal (Decision: {decision})")
                # Still show cost summary
                self.gpt_analyzer.cost_tracker.print_summary()
                return
            
            logger.info(f"🎯 {symbol} - TRADE SIGNAL DETECTED!")
            
            # 9. Extract trade levels
            trade_data = self.extract_trade_levels(analysis)
            trade_data['support'] = technical_data.get('support') if technical_data else None
            trade_data['resistance'] = technical_data.get('resistance') if technical_data else None
            
            # 10. Generate annotated chart
            logger.info(f"📊 Generating chart for {symbol}...")
            chart_buf = self.chart_gen.create_chart_with_annotations(
                candles[-100:],  # Last 100 candles for chart
                symbol,
                spot_price,
                trade_data
            )
            
            if not chart_buf:
                logger.warning(f"⚠️ Chart generation failed for {symbol}")
                return
            
            # 11. Send to Telegram
            logger.info(f"📤 Sending alert to Telegram...")
            
            # Chart first
            chart_buf.seek(0)
            await self.bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=chart_buf,
                caption=f"📊 *{symbol} TRADE SIGNAL*\n💰 Spot: ₹{spot_price:,.2f}\n🎯 Groq Score: {groq_score}/10",
                parse_mode='Markdown'
            )
            
            # Then analysis
            header = f"🤖 GPT O1-MINI TRADE ANALYSIS\n"
            header += f"{'='*50}\n"
            header += f"📊 Symbol: {symbol}\n"
            header += f"💰 Current Price: ₹{spot_price:,.2f}\n"
            header += f"⏰ Time: {datetime.now().strftime('%d-%m-%Y %H:%M IST')}\n"
            header += f"{'='*50}\n\n"
            
            full_message = header + analysis
            
            # Split if too long
            if len(full_message) > 4000:
                parts = [full_message[i:i+4000] for i in range(0, len(full_message), 4000)]
                for idx, part in enumerate(parts, 1):
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=f"[Part {idx}/{len(parts)}]\n\n{part}"
                    )
                    await asyncio.sleep(1)
            else:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=full_message
                )
            
            # Cost summary
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"💰 Analysis Cost: ₹{cost:.2f}\n📊 Total Spent Today: ₹{self.gpt_analyzer.cost_tracker.get_total_cost():.2f}"
            )
            
            logger.info(f"✅ {symbol} alert sent successfully!")
            
            # Show cost summary in logs
            self.gpt_analyzer.cost_tracker.print_summary()
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Trading Bot with Groq + GPT o1-mini...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load security IDs")
            return
        
        await self.send_startup_message()
        
        all_symbols = list(self.security_id_map.keys())
        
        logger.info(f"📊 Total symbols to monitor: {len(all_symbols)}")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 NEW SCAN CYCLE: {timestamp}")
                logger.info(f"{'='*70}\n")
                
                for idx, symbol in enumerate(all_symbols, 1):
                    logger.info(f"📊 [{idx}/{len(all_symbols)}] Processing {symbol}...")
                    
                    await self.analyze_and_send(symbol)
                    
                    # Wait between symbols
                    if idx < len(all_symbols):
                        logger.info(f"⏳ Waiting 10 seconds before next symbol...")
                        await asyncio.sleep(10)
                
                logger.info("\n" + "="*70)
                logger.info("✅ SCAN CYCLE COMPLETED!")
                logger.info(f"💰 Total cost this session: ₹{self.gpt_analyzer.cost_tracker.get_total_cost():.2f}")
                logger.info("⏳ Next cycle in 15 minutes...")
                logger.info("="*70 + "\n")
                
                await asyncio.sleep(900)  # 15 minutes
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup notification"""
        try:
            msg = "🤖 *TRADING BOT ACTIVATED!*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            msg += f"📊 *Tracking:* {len(self.security_id_map)} Stocks/Indices\n"
            msg += f"⏱️ *Scan Frequency:* Every 15 minutes\n"
            msg += f"📈 *Timeframe:* 5-minute candles (360+ candles)\n\n"
            
            msg += "🎯 *STRATEGY:*\n"
            msg += "  1️⃣ Groq Pre-Filter (FREE) → Score 0-10\n"
            msg += "  2️⃣ If score ≥ 6 → GPT o1-mini Deep Analysis\n"
            msg += "  3️⃣ Only send if confidence > 70%\n\n"
            
            msg += "🎯 *FEATURES:*\n"
            msg += "  ✅ 360+ Candlestick Analysis\n"
            msg += "  ✅ Option Chain (PCR, OI, IV, Max Pain)\n"
            msg += "  ✅ Technical Indicators (SMA, RSI)\n"
            msg += "  ✅ Pattern Detection\n"
            msg += "  ✅ AI Deep Reasoning\n"
            msg += "  ✅ Entry/Target/SL Levels\n"
            msg += "  ✅ Risk:Reward Calculation\n"
            msg += "  ✅ Annotated Charts (White BG)\n"
            msg += "  ✅ Conservative Approach\n\n"
            
            msg += "⚡ *POWERED BY:*\n"
            msg += "  • Groq (Llama 3.3 70B) - Filter\n"
            msg += "  • GPT o1-mini - Deep Analysis\n"
            msg += "  • DhanHQ API v2 - Data\n\n"
            
            msg += "📋 *MONITORING:*\n"
            msg += f"  • 2 Indices: NIFTY 50, NIFTY BANK\n"
            msg += f"  • 60 Top Stocks (NIFTY 50 + Top 10)\n\n"
            
            msg += "💰 *COST ESTIMATE:*\n"
            msg += "  • Groq: FREE (unlimited)\n"
            msg += "  • o1-mini: ~₹1.43 per deep analysis\n"
            msg += "  • Expected: ₹30-60/day\n"
            msg += "  • Budget: ₹300/month ✅\n\n"
            
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


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    try:
        # Environment variables check
        required_vars = {
            'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
            'DHAN_CLIENT_ID': DHAN_CLIENT_ID,
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN,
            'GROQ_API_KEY': GROQ_API_KEY,
            'OPENAI_API_KEY': OPENAI_API_KEY
        }
        
        missing_vars = [k for k, v in required_vars.items() if not v]
        
        if missing_vars:
            logger.error("❌ MISSING ENVIRONMENT VARIABLES!")
            logger.error(f"Missing: {', '.join(missing_vars)}")
            logger.error("\n⚙️ Please set these in Railway.app:")
            for var in missing_vars:
                logger.error(f"  - {var}")
            exit(1)
        
        logger.info("✅ All environment variables present")
        logger.info("🚀 Initializing Trading Bot...")
        logger.info("🔍 Filter: Groq (FREE)")
        logger.info("🧠 Analysis: GPT o1-mini (₹1.43 per call)")
        logger.info(f"📊 Monitoring: {len(STOCKS_INDICES)} symbols (2 indices + 60 stocks)")
        
        bot = TradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
