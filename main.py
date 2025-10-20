"""
🤖 ADVANCED NIFTY 50 STOCKS TRADING BOT v6.0 - DEEPSEEK V3 ENHANCED
✅ Last 5 Days Complete Candlestick Analysis
✅ ATM ± 11 Strikes Full Option Chain Analysis
✅ DeepSeek V3 Deep Market Analysis
✅ Proper Chart-Based Entry/Target/SL Calculation
✅ Enhanced Error Handling

Author: Advanced Trading System
Version: 6.0 - DEEPSEEK ENHANCED
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
    SCAN_INTERVAL = 900  # 15 minutes
    CONFIDENCE_THRESHOLD = 75
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 3600
    
    # Enhanced Analysis Settings
    LOOKBACK_DAYS = 5  # Last 5 days data
    ATM_STRIKE_RANGE = 11  # ATM ± 11 strikes
    CANDLE_ANALYSIS_COUNT = 50  # Last 50 candles for pattern detection
    
    # NIFTY 50 Stocks
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
class OIData:
    """Enhanced Option Chain Data Model"""
    strike: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_oi_change: int
    pe_oi_change: int
    ce_iv: float = 0.0
    pe_iv: float = 0.0
    pcr_at_strike: float = 0.0


@dataclass
class CandlePattern:
    """Candlestick Pattern Data"""
    timestamp: str
    candle_type: str  # BULLISH/BEARISH
    body_size: float
    upper_wick: float
    lower_wick: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    significance: str  # STRONG/MODERATE/WEAK


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
        """Store option chain with enhanced OI data"""
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
                        'pe_volume': oi.pe_volume,
                        'ce_iv': oi.ce_iv,
                        'pe_iv': oi.pe_iv
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
        """Enhanced OI comparison with detailed changes"""
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
                
                # Track significant changes (>500 OI change)
                if abs(ce_oi_change) > 500 or abs(pe_oi_change) > 500:
                    deltas.append({
                        'strike': curr_oi.strike,
                        'ce_oi': curr_oi.ce_oi,
                        'pe_oi': curr_oi.pe_oi,
                        'ce_oi_change': ce_oi_change,
                        'pe_oi_change': pe_oi_change,
                        'ce_volume': curr_oi.ce_volume,
                        'pe_volume': curr_oi.pe_volume,
                        'pcr_at_strike': curr_oi.pcr_at_strike
                    })
            
            # Sort by absolute OI change (most significant first)
            deltas.sort(key=lambda x: abs(x['ce_oi_change']) + abs(x['pe_oi_change']), reverse=True)
            
            time_diff = (datetime.now(pytz.timezone('Asia/Kolkata')) - 
                        datetime.fromisoformat(old_data['timestamp'])).seconds / 60
            
            return {
                'change': 'UPDATED',
                'deltas': deltas[:15],  # Top 15 significant changes
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
            
            logger.info(f"📊 Total instruments in CSV: {len(all_rows)}")
            
            for stock_symbol in Config.NIFTY_50_STOCKS:
                found = False
                for row in all_rows:
                    try:
                        trading_symbol = row.get('SEM_TRADING_SYMBOL', '').strip()
                        segment = row.get('SEM_SEGMENT', '').strip()
                        exch_segment = row.get('SEM_EXM_EXCH_ID', '').strip()
                        
                        if (segment == 'E' and 
                            exch_segment == 'NSE' and 
                            trading_symbol == stock_symbol):
                            
                            sec_id = row.get('SEM_SMST_SECURITY_ID', '').strip()
                            if sec_id:
                                self.security_id_map[stock_symbol] = {
                                    'security_id': int(sec_id),
                                    'segment': 'NSE_EQ',
                                    'trading_symbol': trading_symbol,
                                    'instrument': 'EQUITY'
                                }
                                logger.info(f"✅ {stock_symbol}: Security ID = {sec_id}")
                                found = True
                                break
                    except Exception:
                        continue
                
                if not found:
                    logger.warning(f"⚠️ {stock_symbol}: Not found in instruments CSV")
            
            logger.info(f"🎯 Successfully loaded {len(self.security_id_map)}/50 NIFTY stocks")
            return len(self.security_id_map) > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading securities: {e}")
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        """Get nearest expiry for stock options"""
        try:
            payload = {
                "UnderlyingScrip": int(security_id),
                "UnderlyingSeg": "NSE_EQ"
            }
            
            response = requests.post(
                Config.DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data') and len(data['data']) > 0:
                    return data['data'][0]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Expiry error: {e}")
            return None
    
    def get_historical_candles(self, security_id: int, segment: str, symbol: str, 
                              lookback_days: int = 5) -> Optional[pd.DataFrame]:
        """Get last 5 days historical candles"""
        try:
            logger.info(f"📊 Fetching {lookback_days} days candles for {symbol}")
            
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=lookback_days)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "expiryCode": 0,
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
                
                if 'timestamp' in data and 'open' in data and len(data['open']) > 0:
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
                else:
                    logger.warning(f"⚠️ {symbol}: No candle data")
                    return None
            else:
                logger.error(f"❌ {symbol}: API error {response.status_code}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Candle fetch error for {symbol}: {e}")
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str, 
                        symbol: str, spot_price: float) -> Optional[List[OIData]]:
        """Get ATM ± 11 strikes option chain data"""
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
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('data'):
                return None
            
            oc_data = data['data'].get('oc', {})
            
            # Find ATM strike (nearest to spot price)
            strikes = [float(s) for s in oc_data.keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            logger.info(f"📍 {symbol} ATM Strike: {atm_strike} (Spot: {spot_price:.2f})")
            
            # Get ATM ± 11 strikes
            strike_range = Config.ATM_STRIKE_RANGE
            oi_list = []
            
            for strike_str, strike_data in oc_data.items():
                try:
                    strike = float(strike_str)
                    
                    # Filter: Only ATM ± 11 strikes
                    strikes_sorted = sorted(strikes)
                    atm_index = strikes_sorted.index(atm_strike)
                    start_idx = max(0, atm_index - strike_range)
                    end_idx = min(len(strikes_sorted), atm_index + strike_range + 1)
                    valid_strikes = strikes_sorted[start_idx:end_idx]
                    
                    if strike not in valid_strikes:
                        continue
                    
                    ce_data = strike_data.get('ce', {})
                    pe_data = strike_data.get('pe', {})
                    
                    ce_oi = ce_data.get('oi', 0)
                    pe_oi = pe_data.get('oi', 0)
                    
                    pcr = pe_oi / ce_oi if ce_oi > 0 else 0
                    
                    oi_list.append(OIData(
                        strike=strike,
                        ce_oi=ce_oi,
                        pe_oi=pe_oi,
                        ce_volume=ce_data.get('volume', 0),
                        pe_volume=pe_data.get('volume', 0),
                        ce_oi_change=0,
                        pe_oi_change=0,
                        ce_iv=ce_data.get('iv', 0.0),
                        pe_iv=pe_data.get('iv', 0.0),
                        pcr_at_strike=pcr
                    ))
                except Exception:
                    continue
            
            logger.info(f"✅ {symbol}: Fetched {len(oi_list)} strikes (ATM ± {strike_range})")
            return oi_list
            
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None


# ========================
# ENHANCED CHART ANALYZER
# ========================
class EnhancedChartAnalyzer:
    """Deep Chart Pattern & Candlestick Analysis"""
    
    @staticmethod
    def analyze_last_5_days_candles(df: pd.DataFrame) -> List[CandlePattern]:
        """Analyze all candles from last 5 days"""
        patterns = []
        
        for idx, row in df.iterrows():
            body = abs(row['close'] - row['open'])
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            candle_range = row['high'] - row['low']
            
            # Determine candle type
            is_bullish = row['close'] > row['open']
            candle_type = "BULLISH" if is_bullish else "BEARISH"
            
            # Determine significance
            if candle_range > 0:
                body_ratio = body / candle_range
                if body_ratio > 0.7:
                    significance = "STRONG"
                elif body_ratio > 0.4:
                    significance = "MODERATE"
                else:
                    significance = "WEAK"
            else:
                significance = "DOJI"
            
            patterns.append(CandlePattern(
                timestamp=idx.strftime('%Y-%m-%d %H:%M'),
                candle_type=candle_type,
                body_size=body,
                upper_wick=upper_wick,
                lower_wick=lower_wick,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                significance=significance
            ))
        
        return patterns
    
    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        """Identify overall trend using multiple timeframes"""
        if len(df) < 50:
            return "INSUFFICIENT_DATA"
        
        recent = df.tail(100)
        
        # Multiple SMAs
        sma_20 = recent['close'].tail(20).mean()
        sma_50 = recent['close'].tail(50).mean()
        sma_100 = recent['close'].mean()
        current_price = recent['close'].iloc[-1]
        
        # Trend strength
        if current_price > sma_20 > sma_50 > sma_100:
            return "STRONG_UPTREND"
        elif current_price > sma_20 > sma_50:
            return "UPTREND"
        elif current_price < sma_20 < sma_50 < sma_100:
            return "STRONG_DOWNTREND"
        elif current_price < sma_20 < sma_50:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def calculate_dynamic_levels(df: pd.DataFrame) -> Dict:
        """Calculate dynamic support/resistance from price action"""
        if len(df) < 50:
            return {}
        
        recent = df.tail(100)
        current = recent['close'].iloc[-1]
        
        # Recent swing points
        highs = recent['high'].tail(50)
        lows = recent['low'].tail(50)
        
        # Find pivot highs
        resistance_levels = []
        for i in range(5, len(highs) - 5):
            if all(highs.iloc[i] >= highs.iloc[i-j] for j in range(1, 6)) and \
               all(highs.iloc[i] >= highs.iloc[i+j] for j in range(1, 6)):
                resistance_levels.append(highs.iloc[i])
        
        # Find pivot lows
        support_levels = []
        for i in range(5, len(lows) - 5):
            if all(lows.iloc[i] <= lows.iloc[i-j] for j in range(1, 6)) and \
               all(lows.iloc[i] <= lows.iloc[i+j] for j in range(1, 6)):
                support_levels.append(lows.iloc[i])
        
        # Cluster nearby levels
        def cluster(levels):
            if not levels:
                return []
            levels = sorted(levels)
            clustered = []
            current_cluster = [levels[0]]
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.005:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            clustered.append(np.mean(current_cluster))
            return clustered
        
        resistance = cluster(resistance_levels)
        support = cluster(support_levels)
        
        # Filter by relevance (1-8% from current price)
        resistance = [r for r in resistance if 0.01 <= (r - current)/current <= 0.08]
        support = [s for s in support if 0.01 <= (current - s)/current <= 0.08]
        
        return {
            'nearest_support': min(support) if support else current * 0.98,
            'nearest_resistance': min(resistance) if resistance else current * 1.02,
            'all_support': sorted(support)[:3],
            'all_resistance': sorted(resistance)[:3],
            'swing_high': highs.max(),
            'swing_low': lows.min()
        }


# ========================
# DEEPSEEK V3 ENHANCED ANALYZER
# ========================
class DeepSeekV3Analyzer:
    """DeepSeek V3 with Deep Market Analysis"""
    
    @staticmethod
    def create_comprehensive_analysis(symbol: str, spot_price: float, 
                                     candle_patterns: List[CandlePattern],
                                     oi_data: List[OIData], oi_comparison: Dict,
                                     chart_levels: Dict, trend: str) -> Optional[Dict]:
        """Deep analysis with last 5 days candles + ATM ± 11 strikes OI"""
        try:
            logger.info(f"🤖 DeepSeek V3: Deep analysis for {symbol}...")
            
            # Format last 50 candles for AI (detailed)
            last_50 = candle_patterns[-50:] if len(candle_patterns) > 50 else candle_patterns
            
            candles_detail = []
            for i, cp in enumerate(last_50[-20:], 1):  # Last 20 candles detail
                candles_detail.append(
                    f"{i}. {cp.timestamp} | {cp.candle_type} ({cp.significance}) | "
                    f"O:{cp.open:.1f} H:{cp.high:.1f} L:{cp.low:.1f} C:{cp.close:.1f} | "
                    f"Body:{cp.body_size:.1f} UpperWick:{cp.upper_wick:.1f} LowerWick:{cp.lower_wick:.1f} | "
                    f"Vol:{cp.volume:,}"
                )
            
            candles_text = "\n".join(candles_detail)
            
            # Count candle types
            bullish_count = sum(1 for c in last_50 if c.candle_type == "BULLISH")
            bearish_count = sum(1 for c in last_50 if c.candle_type == "BEARISH")
            strong_candles = sum(1 for c in last_50[-10:] if c.significance == "STRONG")
            
            # Format OI data (ATM ± 11 strikes)
            oi_data_sorted = sorted(oi_data, key=lambda x: x.strike)
            atm_strike = min(oi_data, key=lambda x: abs(x.strike - spot_price)).strike
            
            oi_table = []
            for oi in oi_data_sorted:
                marker = " ⭐ ATM" if oi.strike == atm_strike else ""
                oi_table.append(
                    f"Strike {oi.strike}{marker} | "
                    f"CE OI:{oi.ce_oi:,} Vol:{oi.ce_volume:,} IV:{oi.ce_iv:.1f} | "
                    f"PE OI:{oi.pe_oi:,} Vol:{oi.pe_volume:,} IV:{oi.pe_iv:.1f} | "
                    f"PCR:{oi.pcr_at_strike:.2f}"
                )
            
            oi_text = "\n".join(oi_table)
            
            # Format OI changes
            oi_changes_text = "No previous data (First scan)"
            if oi_comparison.get('deltas'):
                changes = []
                for d in oi_comparison['deltas'][:10]:
                    ce_change_pct = (d['ce_oi_change'] / d['ce_oi'] * 100) if d['ce_oi'] > 0 else 0
                    pe_change_pct = (d['pe_oi_change'] / d['pe_oi'] * 100) if d['pe_oi'] > 0 else 0
                    
                    changes.append(
                        f"Strike {d['strike']} | "
                        f"CE: {d['ce_oi_change']:+,} ({ce_change_pct:+.1f}%) OI:{d['ce_oi']:,} | "
                        f"PE: {d['pe_oi_change']:+,} ({pe_change_pct:+.1f}%) OI:{d['pe_oi']:,} | "
                        f"PCR:{d['pcr_at_strike']:.2f}"
                    )
                
                oi_changes_text = "\n".join(changes)
            
            # Calculate PCR
            total_ce_oi = sum(oi.ce_oi for oi in oi_data)
            total_pe_oi = sum(oi.pe_oi for oi in oi_data)
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            # Max Pain
            max_pain = spot_price
            if oi_data:
                pain_values = {}
                for strike_point in [oi.strike for oi in oi_data]:
                    total_pain = 0
                    for oi in oi_data:
                        if oi.strike < strike_point:
                            total_pain += (strike_point - oi.strike) * oi.pe_oi
                        else:
                            total_pain += (oi.strike - strike_point) * oi.ce_oi
                    pain_values[strike_point] = total_pain
                max_pain = min(pain_values, key=pain_values.get)
            
            # DeepSeek V3 API Call
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""You are an expert Indian equity and options trader analyzing {symbol} stock.

═══════════════════════════════════════════════════════════════════
📊 CURRENT MARKET DATA
═══════════════════════════════════════════════════════════════════
Stock: {symbol}
Current Spot Price: Rs {spot_price:.2f}
ATM Strike: {atm_strike}
Trend: {trend}

═══════════════════════════════════════════════════════════════════
📈 CHART LEVELS (Last 5 Days Analysis)
═══════════════════════════════════════════════════════════════════
Swing High: Rs {chart_levels.get('swing_high', spot_price * 1.02):.2f}
Swing Low: Rs {chart_levels.get('swing_low', spot_price * 0.98):.2f}
Nearest Support: Rs {chart_levels.get('nearest_support', spot_price * 0.98):.2f}
Nearest Resistance: Rs {chart_levels.get('nearest_resistance', spot_price * 1.02):.2f}

═══════════════════════════════════════════════════════════════════
🕯️ LAST 20 CANDLESTICKS (15-min timeframe from 5 days data)
═══════════════════════════════════════════════════════════════════
Bullish Candles: {bullish_count}/{len(last_50)} | Bearish: {bearish_count}/{len(last_50)}
Strong Candles (Last 10): {strong_candles}/10

Detailed Candles:
{candles_text}

═══════════════════════════════════════════════════════════════════
⛓️ OPTION CHAIN DATA (ATM ± 11 Strikes)
═══════════════════════════════════════════════════════════════════
Total Strikes Analyzed: {len(oi_data)}
Overall PCR: {pcr:.2f}
Max Pain: Rs {max_pain:.2f}
Total CE OI: {total_ce_oi:,}
Total PE OI: {total_pe_oi:,}

Strike-wise Data:
{oi_text}

═══════════════════════════════════════════════════════════════════
📊 OI CHANGES (vs Previous Scan - Top 10)
═══════════════════════════════════════════════════════════════════
{oi_changes_text}

═══════════════════════════════════════════════════════════════════
🎯 ANALYSIS TASK
═══════════════════════════════════════════════════════════════════

Analyze the above data comprehensively and provide:

1. **PRICE ACTION ANALYSIS**: What is the story told by last 20 candles? Any rejections, breakouts, or consolidations?

2. **OPTION CHAIN INSIGHTS**: 
   - Where is the highest CE OI buildup? (Resistance)
   - Where is the highest PE OI buildup? (Support)
   - Are there significant OI changes showing smart money positioning?
   - Is PCR bullish (>1.2) or bearish (<0.8)?

3. **SUPPORT/RESISTANCE CONFIRMATION**:
   - Does OI data confirm chart support/resistance levels?
   - Is price respecting key levels?

4. **TRADE SETUP**:
   - Should we buy PE (bearish bet) or CE (bullish bet)?
   - Entry at CURRENT SPOT PRICE: Rs {spot_price:.2f}
   - Target based on nearest support/resistance
   - Stop Loss based on swing high/low
   - Which strike to trade? (closest to ATM)

5. **RISK ASSESSMENT**:
   - What is the probability of this trade working?
   - What could go wrong?

═══════════════════════════════════════════════════════════════════
CRITICAL RULES:
═══════════════════════════════════════════════════════════════════
✅ Entry MUST be current spot price: Rs {spot_price:.2f}
✅ Target should be realistic (nearest strong support/resistance)
✅ Stop Loss should be beyond swing high/low
✅ Consider both chart pattern AND option chain signals
✅ If contradicting signals, choose WAIT
✅ Minimum Risk:Reward should be 1:2

═══════════════════════════════════════════════════════════════════
Reply STRICTLY in JSON format (no markdown, no code blocks):
═══════════════════════════════════════════════════════════════════

{{
  "opportunity": "PE_BUY",
  "confidence": 85,
  "recommended_strike": {int(atm_strike)},
  "entry_price": {spot_price:.2f},
  "target": 1380.00,
  "stop_loss": 1410.00,
  "risk_reward": "1:3.5",
  "chart_signal": "Bearish - Multiple rejections at resistance, forming lower highs",
  "oi_signal": "Strong PE buildup at 1400 PE (+15000 OI), CE unwinding at 1420 CE (-8000 OI)",
  "key_levels": "Support: 1380, Resistance: 1410",
  "reasoning": "Price rejected at 1410 resistance 3 times in last 10 candles with increasing selling pressure. OI data confirms bearish setup with PE accumulation at lower strikes and CE unwinding at higher strikes. PCR of 0.65 shows call writers dominating. Entry at current 1395, targeting 1380 support (nearest strong level), SL above 1410 swing high.",
  "probability": "High (75-80%)",
  "risk_factors": "Sudden reversal if breaks above 1410, low volatility may reduce option premiums"
}}

IMPORTANT: Use ACTUAL numbers from the data above. Do NOT use placeholder values."""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert equity and options trader. Analyze chart patterns, candlestick data, and option chain comprehensively. Provide actionable trading signals with proper risk management. Reply in JSON format only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 2000
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            
            if response.status_code != 200:
                logger.error(f"❌ DeepSeek API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ['opportunity', 'confidence', 'entry_price', 'target', 'stop_loss']
                if all(field in analysis for field in required_fields):
                    logger.info(f"✅ DeepSeek V3: {analysis['opportunity']} | "
                              f"Confidence: {analysis['confidence']}% | "
                              f"Entry: {analysis['entry_price']} | "
                              f"Target: {analysis['target']} | "
                              f"SL: {analysis['stop_loss']}")
                    return analysis
                else:
                    logger.warning(f"⚠️ Missing required fields in DeepSeek response")
                    return None
            
            logger.warning("⚠️ Could not parse JSON from DeepSeek response")
            logger.warning(f"Raw response: {content[:500]}")
            return None
            
        except Exception as e:
            logger.error(f"❌ DeepSeek V3 analysis error: {e}")
            logger.error(traceback.format_exc())
            return None


# ========================
# CHART GENERATOR
# ========================
class ChartGenerator:
    """Generate trading charts"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, symbol: str, entry: float, target: float, 
                    stop_loss: float, opportunity: str) -> BytesIO:
        """Create candlestick chart with levels"""
        try:
            logger.info(f"📊 Generating chart for {symbol}")
            
            chart_df = df.tail(50).copy()
            
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
            
            hlines = dict(
                hlines=[entry, target, stop_loss],
                colors=['blue', 'green', 'red'],
                linestyle='--',
                linewidths=2
            )
            
            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style=s,
                title=f"{symbol} - {opportunity}",
                ylabel='Price (₹)',
                volume=False,
                hlines=hlines,
                returnfig=True,
                figsize=(6, 4),
                tight_layout=True
            )
            
            ax = axes[0]
            current_price = chart_df['close'].iloc[-1]
            
            ax.text(len(chart_df), entry, f' Entry: ₹{entry:.2f}', 
                   color='blue', fontweight='bold', va='center', fontsize=8)
            ax.text(len(chart_df), target, f' Target: ₹{target:.2f}', 
                   color='green', fontweight='bold', va='center', fontsize=8)
            ax.text(len(chart_df), stop_loss, f' SL: ₹{stop_loss:.2f}', 
                   color='red', fontweight='bold', va='center', fontsize=8)
            ax.axhline(y=current_price, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.text(len(chart_df), current_price, f' Now: ₹{current_price:.2f}', 
                   color='orange', fontweight='bold', va='center', fontsize=8)
            
            buf = BytesIO()
            fig.savefig(
                buf, 
                format='png', 
                dpi=100,
                bbox_inches='tight', 
                facecolor='white',
                pad_inches=0.05
            )
            buf.seek(0)
            plt.close(fig)
            
            logger.info(f"✅ Chart generated for {symbol}")
            return buf
            
        except Exception as e:
            logger.error(f"❌ Chart generation error: {e}")
            return None


# ========================
# MAIN BOT
# ========================
class AdvancedFOBot:
    """Advanced NIFTY 50 Trading Bot with DeepSeek V3"""
    
    def __init__(self):
        logger.info("🔧 Initializing NIFTY 50 Trading Bot v6.0...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.chart_analyzer = EnhancedChartAnalyzer()
        self.chart_generator = ChartGenerator()
        self.running = True
        logger.info("✅ NIFTY 50 Bot v6.0 initialized")
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        return Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE
    
    def escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return html.escape(str(text))
    
    async def scan_symbol(self, symbol: str, info: Dict):
        """Comprehensive scan with DeepSeek V3 analysis"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 SCANNING: {symbol}")
            logger.info(f"{'='*70}")
            
            # Get expiry
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No F&O available - SKIP")
                return
            
            logger.info(f"📅 Expiry: {expiry}")
            
            # Get last 5 days candles
            candles_df = self.dhan.get_historical_candles(
                security_id, segment, symbol, 
                lookback_days=Config.LOOKBACK_DAYS
            )
            
            if candles_df is None or len(candles_df) < 50:
                logger.warning(f"⚠️ {symbol}: Insufficient candles - SKIP")
                return
            
            spot_price = candles_df['close'].iloc[-1]
            logger.info(f"💰 Spot Price: ₹{spot_price:,.2f}")
            
            # Analyze all candles
            logger.info(f"📈 Analyzing last 5 days candles...")
            candle_patterns = self.chart_analyzer.analyze_last_5_days_candles(candles_df)
            trend = self.chart_analyzer.identify_trend(candles_df)
            chart_levels = self.chart_analyzer.calculate_dynamic_levels(candles_df)
            
            logger.info(f"📊 Trend: {trend}")
            logger.info(f"📍 Chart Levels: Support={chart_levels.get('nearest_support', 0):.2f}, "
                       f"Resistance={chart_levels.get('nearest_resistance', 0):.2f}")
            
            # Get option chain (ATM ± 11 strikes)
            logger.info(f"⛓️ Fetching ATM ± {Config.ATM_STRIKE_RANGE} strikes...")
            oi_data = self.dhan.get_option_chain(security_id, segment, expiry, symbol, spot_price)
            
            if not oi_data or len(oi_data) < 10:
                logger.warning(f"⚠️ {symbol}: Insufficient option data - SKIP")
                return
            
            # Compare with previous OI
            oi_comparison = self.redis.get_oi_comparison(symbol, oi_data)
            self.redis.store_option_chain(symbol, oi_data, spot_price)
            
            logger.info(f"📊 OI Comparison: {len(oi_comparison.get('deltas', []))} significant changes")
            
            # DeepSeek V3 Deep Analysis
            logger.info(f"🤖 Running DeepSeek V3 comprehensive analysis...")
            analysis = DeepSeekV3Analyzer.create_comprehensive_analysis(
                symbol, spot_price, candle_patterns, oi_data, 
                oi_comparison, chart_levels, trend
            )
            
            if not analysis:
                logger.warning(f"⚠️ {symbol}: No AI analysis - SKIP")
                return
            
            # Check confidence
            if analysis['confidence'] < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"⏸️ {symbol}: Low confidence ({analysis['confidence']}%) - NO ALERT")
                return
            
            # Generate chart
            chart_image = self.chart_generator.create_chart(
                candles_df, symbol,
                analysis.get('entry_price', spot_price),
                analysis.get('target', spot_price * 1.03),
                analysis.get('stop_loss', spot_price * 0.97),
                analysis['opportunity']
            )
            
            # Send alert
            alert_sent = await self.send_trading_alert(
                symbol, spot_price, analysis, chart_levels, 
                oi_data, oi_comparison, expiry, chart_image
            )
            
            if alert_sent:
                logger.info(f"✅ {symbol}: ALERT SENT SUCCESSFULLY! 🎉")
            else:
                logger.warning(f"⚠️ {symbol}: Alert sending failed")
            
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_trading_alert(self, symbol: str, spot_price: float, analysis: Dict,
                                chart_levels: Dict, oi_data: List[OIData], 
                                oi_comparison: Dict, expiry: str, chart_image: BytesIO):
        """Send comprehensive trading alert"""
        try:
            # Signal emoji
            signal_map = {
                "PE_BUY": ("🔴", "PE BUY"),
                "CE_BUY": ("🟢", "CE BUY"),
                "WAIT": ("⚪", "WAIT")
            }
            
            signal_emoji, signal_text = signal_map.get(
                analysis['opportunity'], 
                ("⚪", "WAIT")
            )
            
            # Safe HTML escaping
            def safe(val):
                return self.escape_html(val)
            
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
            
            # Calculate R:R
            entry = analysis.get('entry_price', spot_price)
            target = analysis.get('target', spot_price * 1.03)
            sl = analysis.get('stop_loss', spot_price * 0.97)
            risk = abs(entry - sl)
            reward = abs(target - entry)
            rr_ratio = f"1:{reward/risk:.1f}" if risk > 0 else "N/A"
            
            # Calculate PCR
            total_ce = sum(oi.ce_oi for oi in oi_data)
            total_pe = sum(oi.pe_oi for oi in oi_data)
            pcr = total_pe / total_ce if total_ce > 0 else 0
            
            # Short caption for chart
            short_caption = f"""
📊 🚀 <b>{safe(symbol)}</b> {signal_emoji}

<b>{signal_text}</b> | Confidence: {safe(analysis['confidence'])}%

📊 <b>CHART LEVELS (from 5-day data):</b>
Spot: Rs {safe(f'{spot_price:.2f}')}
Entry: Rs {safe(f'{entry:.2f}')}
Target: Rs {safe(f'{target:.2f}')} 
SL: Rs {safe(f'{sl:.2f}')}
Risk:Reward = {safe(rr_ratio)}

📈 Trend: {safe(analysis.get('chart_signal', 'N/A')[:50])} | PCR: {pcr:.2f}
⏰ {ist_time} IST
"""
            
            # Try sending chart
            photo_sent = False
            if chart_image:
                try:
                    await self.bot.send_photo(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        photo=chart_image,
                        caption=short_caption.strip(),
                        parse_mode='HTML'
                    )
                    photo_sent = True
                    logger.info("✅ Chart sent")
                except Exception as e:
                    logger.error(f"❌ Chart upload failed: {e}")
            
            if not photo_sent:
                await self.bot.send_message(
                    chat_id=Config.TELEGRAM_CHAT_ID,
                    text=f"{short_caption.strip()}\n\n⚠️ Chart generation skipped",
                    parse_mode='HTML'
                )
            
            # Detailed analysis message
            support = chart_levels.get('nearest_support', 'N/A')
            resistance = chart_levels.get('nearest_resistance', 'N/A')
            
            # Top OI change
            oi_summary = "First scan"
            if oi_comparison.get('deltas'):
                top = oi_comparison['deltas'][0]
                oi_summary = f"Strike {top['strike']}: CE {top['ce_oi_change']:+,} PE {top['pe_oi_change']:+,}"
            
            detailed = f"""
📊 <b>{safe(symbol)} Analysis</b>

📈 <b>Chart:</b>
Support: Rs{safe(f'{support:.2f}' if isinstance(support, (int, float)) else support)} | Resist: Rs{safe(f'{resistance:.2f}' if isinstance(resistance, (int, float)) else resistance)}
Pattern: {safe(analysis.get('chart_signal', 'N/A')[:80])}

⛓️ <b>Options:</b>
PCR {pcr:.2f} | Expiry: {expiry}
Strike: {safe(analysis.get('recommended_strike', 'N/A'))}

📊 <b>OI Change:</b>
{safe(oi_summary)}

🧠 <b>AI:</b>
{safe(analysis.get('reasoning', 'Analysis')[:300])}

💡 <b>OI Signal:</b>
{safe(analysis.get('oi_signal', 'N/A')[:150])}

⚡ Trade at own risk | DeepSeek V3
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=detailed.strip(),
                parse_mode='HTML'
            )
            
            logger.info("✅ Alert sent successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def send_startup_message(self):
        """Send startup notification"""
        try:
            ist = pytz.timezone('Asia/Kolkata')
            redis_status = "Connected" if self.redis.redis_client else "Disconnected"
            
            msg = f"""
🤖 <b>NIFTY 50 Trading Bot v6.0 Started!</b>

📊 Tracking: <b>50 NIFTY stocks</b>
⏰ Scan Interval: <b>15 minutes</b>
🎯 Confidence: <b>{Config.CONFIDENCE_THRESHOLD}%+</b>
🔴 Redis: {redis_status}

🔍 <b>Enhanced Features:</b>
✅ Last 5 Days Full Candle Analysis
✅ ATM ± 11 Strikes OI Analysis
✅ DeepSeek V3 Deep Market Analysis
✅ Chart Pattern Detection
✅ Proper Entry/Target/SL Calculation
✅ OI Change Tracking
✅ Chart Generation (600x400px)

📈 <b>Loaded: {len(self.dhan.security_id_map)}/50 stocks</b>

🚀 Status: <b>ACTIVE</b> ✅
⏰ {datetime.now(ist).strftime('%Y-%m-%d %H:%M IST')}
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("✅ Startup message sent!")
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("="*70)
        logger.info("🚀 NIFTY 50 BOT v6.0 - DEEPSEEK V3 ENHANCED")
        logger.info("="*70)
        
        # Validate credentials
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            if not getattr(Config, cred):
                missing.append(cred)
        
        if missing:
            logger.error(f"❌ Missing: {', '.join(missing)}")
            return
        
        # Load securities
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("❌ Failed to load securities")
            return
        
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("🎯 Bot RUNNING! Scanning every 15 minutes...")
        logger.info("="*70)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("😴 Market closed. Sleeping...")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 SCAN CYCLE START - {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"{'='*70}")
                
                for idx, (symbol, info) in enumerate(self.dhan.security_id_map.items(), 1):
                    logger.info(f"\n[{idx}/{len(self.dhan.security_id_map)}] {symbol}")
                    await self.scan_symbol(symbol, info)
                    await asyncio.sleep(3)
                
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ CYCLE COMPLETE! Next scan in 15 min")
                logger.info(f"{'='*70}\n")
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                await asyncio.sleep(60)


# ========================
# MAIN
# ========================
async def main():
    """Entry point"""
    try:
        bot = AdvancedFOBot()
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🎬 NIFTY 50 BOT v6.0 - DEEPSEEK V3 ENHANCED STARTING...")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Critical: {e}")
