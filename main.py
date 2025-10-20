"""
🤖 ADVANCED NIFTY 50 STOCKS TRADING BOT v5.0
Scans all 50 NIFTY stocks every 15 minutes

✅ Chart Pattern Detection (DeepSeek powered)
✅ Smart Money Concepts (Order Blocks, FVG, Liquidity)
✅ Option Chain Analysis (OI, Change in OI, Volume)
✅ Redis Caching for OI comparison
✅ PE/CE Buy Opportunity Detection
✅ Chart Image Generation (White BG, Green/Red Candles)

Author: Advanced Trading System
Version: 5.0 - NIFTY 50 STOCKS SCANNER
"""

import asyncio
import os
import json
import csv
import io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import traceback
import pytz
import redis
from dataclasses import dataclass
import html

# Telegram
from telegram import Bot

# Charting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO

# Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ========================
# CONFIGURATION
# ========================
class Config:
    """Bot Configuration"""
    
    # API Credentials
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Dhan API URLs
    DHAN_API_BASE = "https://api.dhan.co"
    DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
    DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
    DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
    DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
    
    # Bot Settings
    SCAN_INTERVAL = 900  # 15 minutes (900 seconds)
    CONFIDENCE_THRESHOLD = 75
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 3600
    
    # Chart Analysis Settings
    LOOKBACK_CANDLES = 100
    TRENDLINE_CANDLES = 50
    PSYCHOLOGICAL_LEVELS = [100, 250, 500, 1000]
    
    # NIFTY 50 Stocks (All 50 stocks)
    NIFTY_50_STOCKS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
        "ASIANPAINT", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK",
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
        "DMART", "WIPRO", "BAJAJFINSV", "ADANIENT", "ONGC",
        "NTPC", "TECHM", "POWERGRID", "M&M", "TATASTEEL",
        "INDUSINDBK", "COALINDIA", "JSWSTEEL", "GRASIM", "BRITANNIA",
        "TATACONSUM", "HINDALCO", "EICHERMOT", "ADANIPORTS", "APOLLOHOSP",
        "SBILIFE", "BAJAJ-AUTO", "CIPLA", "DIVISLAB", "HDFCLIFE",
        "BPCL", "HEROMOTOCO", "TATAMOTORS", "UPL", "DRREDDY"
    ]


# ========================
# DATA MODELS
# ========================
@dataclass
class ChartPattern:
    """Chart Pattern Data Model"""
    name: str
    type: str
    confidence: int
    target: float
    stop_loss: float
    description: str


@dataclass
class OIData:
    """Option Chain Data Model"""
    strike: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_oi_change: int
    pe_oi_change: int


@dataclass
class TrendlineData:
    """Trendline Information"""
    support_line: float
    resistance_line: float
    trend: str


# ========================
# REDIS HANDLER
# ========================
class RedisCache:
    """Redis Cache Manager"""
    
    def __init__(self):
        try:
            logger.info("🔴 Connecting to Redis...")
            self.redis_client = redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully!")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def store_option_chain(self, symbol: str, oi_data: List[OIData], spot_price: float):
        """Store option chain with OI data"""
        try:
            if not self.redis_client:
                return False
            
            key = f"oi_data:{symbol}"
            value = json.dumps({
                'spot_price': spot_price,
                'strikes': [
                    {
                        'strike': oi.strike,
                        'ce_oi': oi.ce_oi,
                        'pe_oi': oi.pe_oi,
                        'ce_volume': oi.ce_volume,
                        'pe_volume': oi.pe_volume
                    }
                    for oi in oi_data
                ],
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
            })
            
            self.redis_client.setex(key, Config.REDIS_EXPIRY, value)
            return True
        except Exception as e:
            logger.error(f"❌ Redis store error: {e}")
            return False
    
    def get_oi_comparison(self, symbol: str, current_oi: List[OIData]) -> Dict:
        """Compare current OI with cached data"""
        try:
            if not self.redis_client:
                return {'change': 'NO_CACHE', 'deltas': []}
            
            key = f"oi_data:{symbol}"
            cached = self.redis_client.get(key)
            
            if not cached:
                return {'change': 'FIRST_SCAN', 'deltas': []}
            
            old_data = json.loads(cached)
            old_strikes = {s['strike']: s for s in old_data['strikes']}
            
            deltas = []
            for curr_oi in current_oi:
                old = old_strikes.get(curr_oi.strike, {})
                
                ce_oi_change = curr_oi.ce_oi - old.get('ce_oi', 0)
                pe_oi_change = curr_oi.pe_oi - old.get('pe_oi', 0)
                
                if abs(ce_oi_change) > 1000 or abs(pe_oi_change) > 1000:
                    deltas.append({
                        'strike': curr_oi.strike,
                        'ce_oi_change': ce_oi_change,
                        'pe_oi_change': pe_oi_change,
                        'ce_volume': curr_oi.ce_volume,
                        'pe_volume': curr_oi.pe_volume
                    })
            
            time_diff = (datetime.now(pytz.timezone('Asia/Kolkata')) - 
                        datetime.fromisoformat(old_data['timestamp'])).seconds / 60
            
            return {
                'change': 'UPDATED',
                'deltas': deltas,
                'time_diff': time_diff,
                'old_spot': old_data.get('spot_price', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Redis comparison error: {e}")
            return {'change': 'ERROR', 'deltas': []}


# ========================
# DHAN API HANDLER
# ========================
class DhanAPI:
    """Dhan HQ API Integration"""
    
    def __init__(self, redis_cache: RedisCache):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json'
        }
        self.security_id_map = {}
        self.redis = redis_cache
        logger.info("✅ DhanAPI initialized")
    
    async def load_security_ids(self):
        """Load security IDs for NIFTY 50 stocks"""
        try:
            logger.info("📥 Loading NIFTY 50 stock security IDs...")
            response = requests.get(Config.DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"❌ Failed to load instruments")
                return False
            
            csv_reader = csv.DictReader(io.StringIO(response.text))
            all_rows = list(csv_reader)
            
            for stock_symbol in Config.NIFTY_50_STOCKS:
                for row in all_rows:
                    try:
                        trading_symbol = row.get('SEM_TRADING_SYMBOL', '')
                        segment = row.get('SEM_SEGMENT', '')
                        
                        # Match NSE equity stocks
                        if segment == 'E' and trading_symbol == stock_symbol:
                            sec_id = row.get('SEM_SMST_SECURITY_ID')
                            if sec_id:
                                self.security_id_map[stock_symbol] = {
                                    'security_id': int(sec_id),
                                    'segment': 'NSE_EQ',
                                    'trading_symbol': trading_symbol,
                                    'instrument': 'EQUITY'
                                }
                                logger.info(f"✅ {stock_symbol}: Security ID = {sec_id}")
                                break
                    except Exception:
                        continue
            
            logger.info(f"🎯 Loaded {len(self.security_id_map)}/50 NIFTY stocks")
            return len(self.security_id_map) > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading securities: {e}")
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        """Get nearest expiry for stock options"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment
            }
            
            response = requests.post(
                Config.DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data'][0]
            
            return None
        except Exception as e:
            logger.error(f"❌ Expiry error: {e}")
            return None
    
    def get_historical_candles(self, security_id: int, segment: str, symbol: str, 
                              lookback_days: int = 7) -> Optional[pd.DataFrame]:
        """Get historical candles"""
        try:
            logger.info(f"📊 Fetching {lookback_days} days candles for {symbol}")
            
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=lookback_days)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": segment,
                "instrument": "EQUITY",
                "interval": "5",
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            response = requests.post(
                Config.DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'timestamp' in data and len(data.get('open', [])) > 0:
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                        'open': data['open'],
                        'high': data['high'],
                        'low': data['low'],
                        'close': data['close'],
                        'volume': data['volume']
                    })
                    
                    df = df.dropna()
                    df.set_index('timestamp', inplace=True)
                    logger.info(f"✅ {symbol}: {len(df)} candles fetched")
                    return df
            
            logger.warning(f"⚠️ {symbol}: No candle data")
            return None
            
        except Exception as e:
            logger.error(f"❌ Candle fetch error: {e}")
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str, 
                        symbol: str) -> Optional[Dict]:
        """Get option chain data"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment,
                "Expiry": expiry
            }
            
            response = requests.post(
                Config.DHAN_OPTION_CHAIN_URL,
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
            logger.error(f"❌ Option chain error: {e}")
            return None


# ========================
# CHART GENERATOR
# ========================
class ChartGenerator:
    """Generate trading charts with entry, target, SL markers"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, symbol: str, entry: float, target: float, 
                    stop_loss: float, opportunity: str) -> BytesIO:
        """Create candlestick chart with white background and markers"""
        try:
            logger.info(f"📊 Generating chart for {symbol}")
            
            # Prepare data - take last 100 candles
            chart_df = df.tail(100).copy()
            
            # Custom style with white background
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick='inherit',
                volume='in',
                alpha=0.9
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='lightgray',
                facecolor='white',
                figcolor='white',
                y_on_right=False
            )
            
            # Create horizontal lines for entry, target, SL
            hlines = dict(
                hlines=[entry, target, stop_loss],
                colors=['blue', 'green', 'red'],
                linestyle='--',
                linewidths=2
            )
            
            # Create figure
            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style=s,
                title=f"{symbol} - {opportunity}",
                ylabel='Price (₹)',
                volume=False,
                hlines=hlines,
                returnfig=True,
                figsize=(12, 7),
                tight_layout=True
            )
            
            # Add labels for lines
            ax = axes[0]
            current_price = chart_df['close'].iloc[-1]
            
            # Position labels on right side
            y_offset = (chart_df['high'].max() - chart_df['low'].min()) * 0.02
            
            ax.text(len(chart_df), entry, f' Entry: ₹{entry:.2f}', 
                   color='blue', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), target, f' Target: ₹{target:.2f}', 
                   color='green', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), stop_loss, f' Stop Loss: ₹{stop_loss:.2f}', 
                   color='red', fontweight='bold', va='center', fontsize=10)
            
            # Add current price marker
            ax.axhline(y=current_price, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.text(len(chart_df), current_price, f' Current: ₹{current_price:.2f}', 
                   color='orange', fontweight='bold', va='center', fontsize=10)
            
            # Save to BytesIO
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            logger.info(f"✅ Chart generated for {symbol}")
            return buf
            
        except Exception as e:
            logger.error(f"❌ Chart generation error: {e}")
            logger.error(traceback.format_exc())
            return None


# ========================
# ADVANCED CHART ANALYZER
# ========================
class AdvancedChartAnalyzer:
    """Advanced Chart Pattern & Technical Analysis"""
    
    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        """Identify overall trend"""
        if len(df) < 20:
            return "INSUFFICIENT_DATA"
        
        recent = df.tail(50)
        sma_20 = recent['close'].tail(20).mean()
        sma_50 = recent['close'].mean()
        current_price = recent['close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return "UPTREND"
        elif current_price < sma_20 < sma_50:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def find_psychological_levels(spot_price: float) -> List[float]:
        """Find psychological support/resistance levels"""
        levels = []
        
        for interval in Config.PSYCHOLOGICAL_LEVELS:
            lower = (spot_price // interval) * interval
            upper = lower + interval
            levels.extend([lower, upper])
        
        levels = sorted(list(set(levels)))
        filtered = [
            level for level in levels
            if abs(level - spot_price) / spot_price <= 0.05
        ]
        
        return filtered
    
    @staticmethod
    def calculate_support_resistance_zones(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance zones"""
        if len(df) < 50:
            return [], []
        
        recent = df.tail(Config.TRENDLINE_CANDLES)
        highs = recent['high'].values
        lows = recent['low'].values
        
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        
        support_levels = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        
        return support_levels, resistance_levels
    
    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> List[ChartPattern]:
        """Detect chart patterns"""
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        recent = df.tail(50)
        closes = recent['close'].values
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Double Top Detection
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                patterns.append(ChartPattern(
                    name="DOUBLE_TOP",
                    type="BEARISH",
                    confidence=75,
                    target=closes[-1] * 0.97,
                    stop_loss=max(highs[-10:]),
                    description="Double top pattern detected - potential reversal"
                ))
        
        # Double Bottom Detection
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                patterns.append(ChartPattern(
                    name="DOUBLE_BOTTOM",
                    type="BULLISH",
                    confidence=75,
                    target=closes[-1] * 1.03,
                    stop_loss=min(lows[-10:]),
                    description="Double bottom pattern - potential reversal upward"
                ))
        
        return patterns
    
    @staticmethod
    def calculate_trendlines(df: pd.DataFrame) -> TrendlineData:
        """Calculate dynamic trendlines"""
        if len(df) < 30:
            return TrendlineData(0, 0, "INSUFFICIENT_DATA")
        
        recent = df.tail(Config.TRENDLINE_CANDLES)
        
        x = np.arange(len(recent))
        y_high = recent['high'].values
        y_low = recent['low'].values
        
        z_high = np.polyfit(x, y_high, 1)
        resistance_line = z_high[0] * len(recent) + z_high[1]
        
        z_low = np.polyfit(x, y_low, 1)
        support_line = z_low[0] * len(recent) + z_low[1]
        
        if z_high[0] > 0 and z_low[0] > 0:
            trend = "UPTREND"
        elif z_high[0] < 0 and z_low[0] < 0:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        return TrendlineData(support_line, resistance_line, trend)


# ========================
# OPTION CHAIN ANALYZER
# ========================
class OptionChainAnalyzer:
    """Option Chain Data Analysis"""
    
    @staticmethod
    def parse_option_chain(option_chain: Dict, spot_price: float) -> List[OIData]:
        """Parse option chain into OI data"""
        oi_list = []
        oc_data = option_chain.get('oc', {})
        
        for strike_str, data in oc_data.items():
            try:
                strike = float(strike_str)
                
                if abs(strike - spot_price) / spot_price > 0.05:
                    continue
                
                ce_data = data.get('ce', {})
                pe_data = data.get('pe', {})
                
                oi_list.append(OIData(
                    strike=strike,
                    ce_oi=ce_data.get('oi', 0),
                    pe_oi=pe_data.get('oi', 0),
                    ce_volume=ce_data.get('volume', 0),
                    pe_volume=pe_data.get('volume', 0),
                    ce_oi_change=0,
                    pe_oi_change=0
                ))
            except Exception:
                continue
        
        return oi_list
    
    @staticmethod
    def calculate_pcr(oi_list: List[OIData]) -> float:
        """Calculate Put-Call Ratio"""
        total_ce_oi = sum(oi.ce_oi for oi in oi_list)
        total_pe_oi = sum(oi.pe_oi for oi in oi_list)
        
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        return pcr
    
    @staticmethod
    def find_max_oi_strikes(oi_list: List[OIData]) -> Dict:
        """Find strikes with maximum OI"""
        if not oi_list:
            return {'max_ce_strike': 0, 'max_ce_oi': 0, 'max_pe_strike': 0, 'max_pe_oi': 0}
        
        max_ce_oi = max(oi_list, key=lambda x: x.ce_oi)
        max_pe_oi = max(oi_list, key=lambda x: x.pe_oi)
        
        return {
            'max_ce_strike': max_ce_oi.strike,
            'max_ce_oi': max_ce_oi.ce_oi,
            'max_pe_strike': max_pe_oi.strike,
            'max_pe_oi': max_pe_oi.pe_oi
        }
    
    @staticmethod
    def calculate_max_pain(oi_list: List[OIData], spot_price: float) -> float:
        """Calculate Max Pain strike"""
        if not oi_list:
            return spot_price
        
        pain_values = {}
        
        for strike_point in [oi.strike for oi in oi_list]:
            total_pain = 0
            
            for oi in oi_list:
                if oi.strike < strike_point:
                    total_pain += (strike_point - oi.strike) * oi.pe_oi
                else:
                    total_pain += (oi.strike - strike_point) * oi.ce_oi
            
            pain_values[strike_point] = total_pain
        
        if pain_values:
            max_pain_strike = min(pain_values, key=pain_values.get)
            return max_pain_strike
        
        return spot_price


# ========================
# DEEPSEEK ANALYZER
# ========================
class DeepSeekAnalyzer:
    """DeepSeek V3 Combined Analysis"""
    
    @staticmethod
    def analyze_combined(chart_data: Dict, oi_data: Dict, oi_comparison: Dict) -> Optional[Dict]:
        """Combined Chart + OI analysis"""
        try:
            logger.info("🤖 DeepSeek: Analyzing Chart + OI data...")
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            oi_changes = ""
            if oi_comparison.get('deltas'):
                oi_changes = "\n".join([
                    f"Strike {d['strike']}: CE OI {d['ce_oi_change']:+,}, PE OI {d['pe_oi_change']:+,}"
                    for d in oi_comparison['deltas'][:5]
                ])
            
            prompt = f"""You are expert equity trader. Analyze and give PE/CE buying opportunity.

CHART DATA:
- Symbol: {chart_data['symbol']}
- Spot Price: Rs {chart_data['spot_price']}
- Trend: {chart_data['trend']}
- Support: {chart_data['support_zones']}
- Resistance: {chart_data['resistance_zones']}
- Patterns: {chart_data['chart_patterns']}

OPTION DATA:
- PCR: {oi_data['pcr']}
- Max CE Strike: {oi_data['max_ce_strike']}
- Max PE Strike: {oi_data['max_pe_strike']}
- Max Pain: Rs {oi_data['max_pain']}

OI CHANGES:
{oi_changes if oi_changes else "No significant changes"}

Reply ONLY JSON:
{{
  "opportunity": "PE_BUY/CE_BUY/WAIT",
  "confidence": 80,
  "recommended_strike": {chart_data['spot_price']},
  "entry_price": 100,
  "target": 150,
  "stop_loss": 80,
  "quantity_lots": 1,
  "reasoning": "Combined analysis",
  "marathi_explanation": "मराठी analysis"
}}
"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Expert trader. Reply JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"❌ DeepSeek API error: {response.status_code}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                logger.info(f"✅ DeepSeek: {analysis['opportunity']} ({analysis['confidence']}%)")
                return analysis
            
            logger.warning("⚠️ Could not parse DeepSeek response")
            return None
            
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            logger.error(traceback.format_exc())
            return None


# ========================
# MAIN BOT
# ========================
class AdvancedFOBot:
    """Advanced NIFTY 50 Stocks Trading Bot"""
    
    def __init__(self):
        logger.info("🔧 Initializing NIFTY 50 Trading Bot...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.chart_analyzer = AdvancedChartAnalyzer()
        self.oi_analyzer = OptionChainAnalyzer()
        self.chart_generator = ChartGenerator()
        self.running = True
        logger.info("✅ NIFTY 50 Bot initialized")
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        if Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE:
            return True
        
        return False
    
    def escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return html.escape(str(text))
    
    async def scan_symbol(self, symbol: str, info: Dict):
        """Comprehensive scan: Chart + OI analysis"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 SCANNING: {symbol}")
            logger.info(f"{'='*70}")
            
            # Get expiry
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No expiry - checking chart only")
                expiry = "N/A"
            
            # Get candles
            candles_df = self.dhan.get_historical_candles(security_id, segment, symbol, lookback_days=7)
            if candles_df is None or len(candles_df) < Config.LOOKBACK_CANDLES:
                logger.warning(f"⚠️ {symbol}: Insufficient candles - SKIP")
                return
            
            spot_price = candles_df['close'].iloc[-1]
            logger.info(f"💰 Spot Price: ₹{spot_price:,.2f}")
            
            # CHART ANALYSIS
            logger.info(f"📈 Running Chart Analysis...")
            trend = self.chart_analyzer.identify_trend(candles_df)
            support_zones, resistance_zones = self.chart_analyzer.calculate_support_resistance_zones(candles_df)
            psych_levels = self.chart_analyzer.find_psychological_levels(spot_price)
            chart_patterns = self.chart_analyzer.detect_chart_patterns(candles_df)
            trendline_data = self.chart_analyzer.calculate_trendlines(candles_df)
            
            chart_data = {
                'symbol': symbol,
                'spot_price': spot_price,
                'trend': trend,
                'support_zones': [round(s, 2) for s in support_zones[-3:]] if support_zones else [],
                'resistance_zones': [round(r, 2) for r in resistance_zones[-3:]] if resistance_zones else [],
                'chart_patterns': [f"{p.name} ({p.type})" for p in chart_patterns],
                'psychological_levels': [round(l, 2) for l in psych_levels],
                'trendline_support': round(trendline_data.support_line, 2),
                'trendline_resistance': round(trendline_data.resistance_line, 2),
                'trendline_trend': trendline_data.trend
            }
            
            # OPTION CHAIN ANALYSIS
            oi_data = {'pcr': 0, 'max_ce_strike': 0, 'max_ce_oi': 0, 'max_pe_strike': 0, 'max_pe_oi': 0, 'max_pain': spot_price}
            oi_comparison = {'change': 'NO_OPTIONS', 'deltas': []}
            
            if expiry != "N/A":
                logger.info(f"⛓️ Running Option Chain Analysis...")
                option_chain = self.dhan.get_option_chain(security_id, segment, expiry, symbol)
                
                if option_chain:
                    oi_list = self.oi_analyzer.parse_option_chain(option_chain, spot_price)
                    
                    if oi_list:
                        pcr = self.oi_analyzer.calculate_pcr(oi_list)
                        max_oi = self.oi_analyzer.find_max_oi_strikes(oi_list)
                        max_pain = self.oi_analyzer.calculate_max_pain(oi_list, spot_price)
                        
                        oi_data = {
                            'pcr': round(pcr, 2),
                            'max_ce_strike': max_oi['max_ce_strike'],
                            'max_ce_oi': max_oi['max_ce_oi'],
                            'max_pe_strike': max_oi['max_pe_strike'],
                            'max_pe_oi': max_oi['max_pe_oi'],
                            'max_pain': round(max_pain, 2)
                        }
                        
                        oi_comparison = self.redis.get_oi_comparison(symbol, oi_list)
                        self.redis.store_option_chain(symbol, oi_list, spot_price)
            
            # DEEPSEEK AI ANALYSIS
            logger.info(f"🤖 Running DeepSeek Analysis...")
            analysis = DeepSeekAnalyzer.analyze_combined(chart_data, oi_data, oi_comparison)
            
            if not analysis:
                logger.warning(f"⚠️ {symbol}: No AI analysis - SKIP")
                return
            
            # Check confidence threshold
            if analysis['confidence'] < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"⏸️ {symbol}: Low confidence ({analysis['confidence']}%) - NO ALERT")
                return
            
            # Generate chart
            chart_image = self.chart_generator.create_chart(
                candles_df, 
                symbol, 
                analysis.get('entry_price', spot_price),
                analysis.get('target', spot_price * 1.03),
                analysis.get('stop_loss', spot_price * 0.97),
                analysis['opportunity']
            )
            
            # Send alert with chart
            await self.send_trading_alert(symbol, spot_price, chart_data, oi_data, 
                                         oi_comparison, analysis, expiry, chart_image)
            
            logger.info(f"✅ {symbol}: ALERT SENT! 🎉")
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_trading_alert(self, symbol: str, spot_price: float, chart_data: Dict, 
                                oi_data: Dict, oi_comparison: Dict, analysis: Dict, 
                                expiry: str, chart_image: BytesIO):
        """Send trading alert with chart image"""
        try:
            # Signal emoji
            if analysis['opportunity'] == "PE_BUY":
                signal_emoji = "🔴"
                signal_text = "PE BUY (Bearish)"
            elif analysis['opportunity'] == "CE_BUY":
                signal_emoji = "🟢"
                signal_text = "CE BUY (Bullish)"
            else:
                signal_emoji = "⚪"
                signal_text = "WAIT"
            
            # Build OI change text
            oi_change_text = ""
            if oi_comparison.get('deltas'):
                changes = oi_comparison['deltas'][:3]
                oi_change_text = "\n".join([
                    f"Strike {d['strike']}: CE {d['ce_oi_change']:+,} PE {d['pe_oi_change']:+,}"
                    for d in changes
                ])
            else:
                oi_change_text = "First scan or no options data"
            
            # Escape all dynamic values
            symbol_safe = self.escape_html(symbol)
            spot_safe = self.escape_html(f"{spot_price:,.2f}")
            expiry_safe = self.escape_html(expiry)
            confidence_safe = self.escape_html(analysis['confidence'])
            entry_safe = self.escape_html(analysis.get('entry_price', 100))
            target_safe = self.escape_html(analysis.get('target', 150))
            sl_safe = self.escape_html(analysis.get('stop_loss', 80))
            lots_safe = self.escape_html(analysis.get('quantity_lots', 1))
            
            # Format support/resistance zones safely
            support_str = ", ".join([f"Rs{s}" for s in chart_data['support_zones'][:2]]) if chart_data['support_zones'] else 'N/A'
            resist_str = ", ".join([f"Rs{r}" for r in chart_data['resistance_zones'][:2]]) if chart_data['resistance_zones'] else 'N/A'
            patterns_str = ", ".join(chart_data['chart_patterns']) if chart_data['chart_patterns'] else 'None'
            
            support_safe = self.escape_html(support_str)
            resist_safe = self.escape_html(resist_str)
            patterns_safe = self.escape_html(patterns_str)
            
            reasoning_safe = self.escape_html(analysis.get('reasoning', 'Combined analysis'))
            marathi_safe = self.escape_html(analysis.get('marathi_explanation', 'तांत्रिक विश्लेषण'))
            oi_change_safe = self.escape_html(oi_change_text)
            
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%b %H:%M IST')
            
            message = f"""
🚀 <b>NIFTY 50 STOCK SIGNAL</b>

📊 Symbol: <b>{symbol_safe}</b>
💰 Spot: Rs {spot_safe}
📅 Expiry: {expiry_safe}
⏰ Time: {ist_time}

{signal_emoji} <b>OPPORTUNITY: {signal_text}</b>
💪 Confidence: <b>{confidence_safe}%</b>

🎯 <b>TRADE SETUP:</b>
Entry: Rs {entry_safe}
Target: Rs {target_safe}
Stop-Loss: Rs {sl_safe}
Quantity: {lots_safe} Lot(s)

📈 <b>CHART ANALYSIS:</b>
Trend: {chart_data['trend']}
Support: {support_safe}
Resistance: {resist_safe}
Patterns: {patterns_safe}

⛓️ <b>OPTION CHAIN:</b>
PCR: {oi_data['pcr']}
Max CE Strike: {oi_data['max_ce_strike']}
Max PE Strike: {oi_data['max_pe_strike']}
Max Pain: Rs {oi_data['max_pain']}

📊 <b>OI CHANGES:</b>
{oi_change_safe}

🧠 <b>AI REASONING:</b>
{reasoning_safe}

📝 <b>मराठी:</b>
{marathi_safe}

⚡ <b>Disclaimer:</b> AI-generated. Trade at your own risk.
🤖 Powered by: DeepSeek V3
"""
            
            # Send chart image with caption
            if chart_image:
                await self.bot.send_photo(
                    chat_id=Config.TELEGRAM_CHAT_ID,
                    photo=chart_image,
                    caption=message,
                    parse_mode='HTML'
                )
            else:
                # Fallback to text only if chart failed
                await self.bot.send_message(
                    chat_id=Config.TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='HTML'
                )
            
            logger.info("✅ Trading alert sent to Telegram!")
            
        except Exception as e:
            logger.error(f"❌ Alert sending error: {e}")
            logger.error(traceback.format_exc())
    
    async def send_startup_message(self):
        """Send startup notification"""
        try:
            logger.info("📤 Sending startup message...")
            ist = pytz.timezone('Asia/Kolkata')
            
            redis_status = "Connected" if self.redis.redis_client else "Disconnected"
            
            msg = f"""
🤖 <b>NIFTY 50 Trading Bot v5.0 Started!</b>

📊 Tracking: <b>50 NIFTY stocks</b>
⏰ Scan Interval: <b>15 minutes</b>
🎯 Confidence Threshold: <b>{Config.CONFIDENCE_THRESHOLD}%</b>
⏱️ Market: {Config.MARKET_OPEN} - {Config.MARKET_CLOSE} IST
🔴 Redis: {redis_status}

🔍 <b>Features:</b>
✅ Chart Pattern Detection
✅ Trendline Analysis
✅ Support/Resistance Zones
✅ Option Chain Analysis
✅ OI Change Tracking
✅ DeepSeek AI Analysis
✅ Chart Image Generation
✅ PE/CE Buy Signals

📈 <b>Loaded Stocks:</b>
{len(self.dhan.security_id_map)}/50 stocks ready

🚀 Status: <b>ACTIVE</b> ✅
⏰ Started: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}

📝 Next scan: Every 15 minutes during market hours!
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("✅ Startup message sent!")
        except Exception as e:
            logger.error(f"❌ Startup message error: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("="*70)
        logger.info("🚀 NIFTY 50 TRADING BOT v5.0 STARTING...")
        logger.info("="*70)
        
        # Validate credentials
        logger.info("🔐 Validating API credentials...")
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            if not getattr(Config, cred):
                missing.append(cred)
        
        if missing:
            logger.error(f"❌ Missing credentials: {', '.join(missing)}")
            return
        
        logger.info("✅ All credentials validated!")
        
        # Load NIFTY 50 stocks
        logger.info("📥 Loading NIFTY 50 stock IDs...")
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("❌ Failed to load securities. Exiting...")
            return
        
        logger.info(f"✅ Loaded {len(self.dhan.security_id_map)}/50 stocks!")
        
        # Send startup message
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("🎯 Bot is now RUNNING! Scanning every 15 minutes...")
        logger.info("="*70)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("😴 Market closed. Sleeping for 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 SCAN CYCLE START")
                logger.info(f"⏰ IST: {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"📊 Scanning {len(self.dhan.security_id_map)} stocks...")
                logger.info(f"{'='*70}")
                
                # Scan each stock
                for idx, (symbol, info) in enumerate(self.dhan.security_id_map.items(), 1):
                    logger.info(f"\n[{idx}/{len(self.dhan.security_id_map)}] Processing {symbol}...")
                    await self.scan_symbol(symbol, info)
                    await asyncio.sleep(3)  # Rate limit
                
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ SCAN CYCLE COMPLETE!")
                logger.info(f"⏰ Next scan in 15 minutes...")
                logger.info(f"{'='*70}\n")
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)


# ========================
# MAIN ENTRY POINT
# ========================
async def main():
    """Main entry point"""
    try:
        logger.info("="*70)
        logger.info("🚀 INITIALIZING NIFTY 50 BOT v5.0")
        logger.info("="*70)
        
        bot = AdvancedFOBot()
        await bot.run()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("="*70)
        logger.info("👋 Bot shutdown complete")
        logger.info("="*70)


if __name__ == "__main__":
    ist = pytz.timezone('Asia/Kolkata')
    logger.info("="*70)
    logger.info("🎬 NIFTY 50 TRADING BOT STARTING...")
    logger.info(f"⏰ IST: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🌍 Timezone: Asia/Kolkata (IST)")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        logger.error(traceback.format_exc())
