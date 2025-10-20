"""
🤖 SMART MONEY F&O TRADING BOT v3.0
Complete integration: Dhan API + DeepSeek AI + Redis Caching

✅ Redis: Option chain data caching & comparison
✅ All major stocks tracking (15+ symbols)
✅ Historical data comparison
✅ Detailed deploy logs at every step
✅ IST timezone support

Author: Trading Bot Team
Version: 3.0 - PRODUCTION READY
"""

import asyncio
import os
import time
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

# Matplotlib imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf

# Telegram
from telegram import Bot

# Logging setup
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
    SCAN_INTERVAL = 300  # 5 minutes
    CONFIDENCE_THRESHOLD = 70  # Minimum confidence for alert
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 3600  # 1 hour cache
    
    # Stocks/Indices to track
    SYMBOLS = {
        # Indices - Try multiple variations
        "NIFTY": {"symbol": "NIFTY 50", "segment": "IDX_I", "alternatives": ["Nifty 50", "NIFTY50", "NIFTY"]},
        "BANKNIFTY": {"symbol": "NIFTY BANK", "segment": "IDX_I", "alternatives": ["Nifty Bank", "NIFTYBANK", "BANKNIFTY"]},
        
        # Top Stocks
        "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
        "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
        "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
        "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
        "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
        "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ"},
        "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ"},
        "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
        "LTIM": {"symbol": "LTIM", "segment": "NSE_EQ"},
        "ADANIENT": {"symbol": "ADANIENT", "segment": "NSE_EQ"},
        "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ"},
        "LT": {"symbol": "LT", "segment": "NSE_EQ"},
        "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ"},
    }


# ========================
# REDIS HANDLER
# ========================
class RedisCache:
    """Redis Cache Manager for Option Chain Data"""
    
    def __init__(self):
        try:
            logger.info("🔴 Connecting to Redis...")
            self.redis_client = redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully!")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def set_option_chain(self, symbol: str, data: Dict):
        """Store option chain data in Redis"""
        try:
            if not self.redis_client:
                return False
            
            key = f"option_chain:{symbol}"
            value = json.dumps({
                'data': data,
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
            })
            
            self.redis_client.setex(key, Config.REDIS_EXPIRY, value)
            logger.info(f"💾 Redis: Stored option chain for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Redis set error: {e}")
            return False
    
    def get_option_chain(self, symbol: str) -> Optional[Dict]:
        """Retrieve option chain data from Redis"""
        try:
            if not self.redis_client:
                return None
            
            key = f"option_chain:{symbol}"
            value = self.redis_client.get(key)
            
            if value:
                cached_data = json.loads(value)
                logger.info(f"📦 Redis: Retrieved cached data for {symbol} (stored: {cached_data['timestamp']})")
                return cached_data
            
            logger.info(f"⚠️ Redis: No cached data for {symbol}")
            return None
        except Exception as e:
            logger.error(f"❌ Redis get error: {e}")
            return None
    
    def compare_option_chain(self, symbol: str, current_data: Dict) -> Dict:
        """Compare current OI data with cached data"""
        try:
            cached = self.get_option_chain(symbol)
            
            if not cached:
                logger.info(f"📊 {symbol}: No previous data for comparison")
                return {'change': 'FIRST_SCAN', 'delta': {}}
            
            old_data = cached['data']
            old_oc = old_data.get('oc', {})
            new_oc = current_data.get('oc', {})
            
            # Calculate OI changes
            total_call_oi_old = sum(strike.get('ce', {}).get('oi', 0) for strike in old_oc.values())
            total_put_oi_old = sum(strike.get('pe', {}).get('oi', 0) for strike in old_oc.values())
            
            total_call_oi_new = sum(strike.get('ce', {}).get('oi', 0) for strike in new_oc.values())
            total_put_oi_new = sum(strike.get('pe', {}).get('oi', 0) for strike in new_oc.values())
            
            call_oi_change = total_call_oi_new - total_call_oi_old
            put_oi_change = total_put_oi_new - total_put_oi_old
            
            pcr_old = total_put_oi_old / total_call_oi_old if total_call_oi_old > 0 else 0
            pcr_new = total_put_oi_new / total_call_oi_new if total_call_oi_new > 0 else 0
            pcr_change = pcr_new - pcr_old
            
            result = {
                'change': 'UPDATED',
                'delta': {
                    'call_oi_change': call_oi_change,
                    'put_oi_change': put_oi_change,
                    'pcr_old': pcr_old,
                    'pcr_new': pcr_new,
                    'pcr_change': pcr_change,
                    'time_diff': (datetime.now(pytz.timezone('Asia/Kolkata')) - 
                                 datetime.fromisoformat(cached['timestamp'])).seconds / 60
                }
            }
            
            logger.info(f"📊 {symbol} OI Comparison:")
            logger.info(f"   Call OI: {call_oi_change:+,.0f} | Put OI: {put_oi_change:+,.0f}")
            logger.info(f"   PCR: {pcr_old:.2f} → {pcr_new:.2f} ({pcr_change:+.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Redis compare error: {e}")
            return {'change': 'ERROR', 'delta': {}}


# ========================
# DHAN API HANDLER
# ========================
class DhanAPI:
    """Dhan HQ API Integration"""
    
    def __init__(self, redis_cache: RedisCache):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.security_id_map = {}
        self.redis = redis_cache
        logger.info("✅ DhanAPI initialized")
    
    async def load_security_ids(self):
        """Load security IDs from Dhan CSV"""
        try:
            logger.info("📥 Loading security IDs from Dhan CSV...")
            response = requests.get(Config.DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"❌ Failed to load instruments: HTTP {response.status_code}")
                return False
            
            logger.info(f"✅ CSV downloaded successfully ({len(response.text)} bytes)")
            
            csv_reader = csv.DictReader(io.StringIO(response.text))
            all_rows = list(csv_reader)
            
            logger.info(f"📊 Total CSV rows: {len(all_rows)}")
            
            for symbol, info in Config.SYMBOLS.items():
                segment = info['segment']
                symbol_name = info['symbol']
                alternatives = info.get('alternatives', [symbol_name])
                
                logger.info(f"🔍 Looking for: {symbol} (trying: {alternatives})")
                
                found = False
                for row in all_rows:
                    try:
                        if segment == "IDX_I":
                            # Try all alternative names
                            trading_symbol = row.get('SEM_TRADING_SYMBOL', '')
                            if (row.get('SEM_SEGMENT') == 'I' and 
                                trading_symbol in alternatives):
                                sec_id = row.get('SEM_SMST_SECURITY_ID')
                                if sec_id:
                                    self.security_id_map[symbol] = {
                                        'security_id': int(sec_id),
                                        'segment': segment,
                                        'trading_symbol': trading_symbol
                                    }
                                    logger.info(f"✅ {symbol}: Security ID = {sec_id} (matched: {trading_symbol})")
                                    found = True
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
                                    logger.info(f"✅ {symbol}: Security ID = {sec_id}")
                                    found = True
                                    break
                    except Exception:
                        continue
                
                if not found:
                    logger.warning(f"⚠️ {symbol}: NOT FOUND in CSV (tried: {alternatives})")
            
            logger.info(f"🎯 Total {len(self.security_id_map)}/{len(Config.SYMBOLS)} securities loaded!")
            return len(self.security_id_map) > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading security IDs: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        """Get nearest expiry for options"""
        try:
            logger.info(f"📅 Getting expiry for security_id={security_id}")
            
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
                if data.get('status') == 'success' and data.get('data'):
                    expiries = data['data']
                    if expiries:
                        nearest = expiries[0]
                        logger.info(f"✅ Nearest expiry: {nearest}")
                        return nearest
            
            logger.warning("⚠️ No expiry found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting expiry: {e}")
            return None
    
    def get_historical_candles(self, security_id: int, segment: str, symbol: str) -> Optional[pd.DataFrame]:
        """Get last 5 days of 5-minute candles"""
        try:
            logger.info(f"📊 Fetching candles for {symbol}")
            
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
            to_date = datetime.now(pytz.timezone('Asia/Kolkata'))
            from_date = to_date - timedelta(days=7)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exch_seg,
                "instrument": instrument,
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
                
                # Check if data has required keys
                if 'open' in data and 'close' in data:
                    # Get all arrays
                    timestamps = data.get('start_Time', [])
                    opens = data.get('open', [])
                    highs = data.get('high', [])
                    lows = data.get('low', [])
                    closes = data.get('close', [])
                    volumes = data.get('volume', [])
                    
                    # Find minimum length (Dhan API sometimes returns mismatched arrays)
                    min_length = min(
                        len(timestamps),
                        len(opens),
                        len(highs),
                        len(lows),
                        len(closes),
                        len(volumes)
                    )
                    
                    if min_length == 0:
                        logger.warning(f"⚠️ {symbol}: Empty arrays in API response")
                        return None
                    
                    logger.info(f"📊 {symbol}: Array lengths - timestamps:{len(timestamps)}, OHLC:{len(opens)}, using min:{min_length}")
                    
                    # Trim all arrays to same length
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(timestamps[:min_length]),
                        'open': opens[:min_length],
                        'high': highs[:min_length],
                        'low': lows[:min_length],
                        'close': closes[:min_length],
                        'volume': volumes[:min_length]
                    })
                    
                    # Remove any NaN rows
                    df = df.dropna()
                    
                    logger.info(f"✅ {symbol}: Fetched {len(df)} clean candles")
                    return df
            
            logger.warning(f"⚠️ {symbol}: No candle data (HTTP {response.status_code})")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching candles for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str, symbol: str) -> Optional[Dict]:
        """Get option chain data with Redis caching"""
        try:
            logger.info(f"⛓️ Fetching option chain for {symbol}")
            
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
                    option_chain = data['data']
                    
                    # Compare with Redis cache
                    comparison = self.redis.compare_option_chain(symbol, option_chain)
                    
                    # Store new data in Redis
                    self.redis.set_option_chain(symbol, option_chain)
                    
                    logger.info(f"✅ Option chain data received for {symbol}")
                    
                    # Add comparison data to result
                    option_chain['_redis_comparison'] = comparison
                    
                    return option_chain
            
            logger.warning(f"⚠️ No option chain data for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting option chain: {e}")
            return None


# ========================
# PATTERN DETECTOR
# ========================
class PatternDetector:
    """Candlestick Pattern Detection"""
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[Dict]:
        """Detect all patterns"""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        def analyze_candle(candle):
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            return {
                'body': body,
                'range': total_range if total_range > 0 else 0.01,
                'upper_wick': upper_wick,
                'lower_wick': lower_wick,
                'is_bullish': candle['close'] > candle['open']
            }
        
        current = analyze_candle(c3)
        prev = analyze_candle(c2)
        
        # HAMMER
        if current['range'] > 0:
            if (current['lower_wick'] > current['body'] * 2 and 
                current['upper_wick'] < current['body'] * 0.5):
                patterns.append({'name': 'HAMMER', 'type': 'BULLISH', 'confidence': 70})
        
        # SHOOTING STAR
        if current['range'] > 0:
            if (current['upper_wick'] > current['body'] * 2 and 
                current['lower_wick'] < current['body'] * 0.5):
                patterns.append({'name': 'SHOOTING_STAR', 'type': 'BEARISH', 'confidence': 70})
        
        # BULLISH ENGULFING
        if (not prev['is_bullish'] and current['is_bullish'] and
            c3['close'] > c2['open'] and c3['open'] < c2['close']):
            patterns.append({'name': 'BULLISH_ENGULFING', 'type': 'BULLISH', 'confidence': 80})
        
        # BEARISH ENGULFING
        if (prev['is_bullish'] and not current['is_bullish'] and
            c3['close'] < c2['open'] and c3['open'] > c2['close']):
            patterns.append({'name': 'BEARISH_ENGULFING', 'type': 'BEARISH', 'confidence': 80})
        
        # DOJI
        if current['range'] > 0 and current['body'] < current['range'] * 0.1:
            patterns.append({'name': 'DOJI', 'type': 'NEUTRAL', 'confidence': 60})
        
        logger.info(f"🔍 Patterns: {len(patterns)} - {[p['name'] for p in patterns]}")
        return patterns


# ========================
# SMART MONEY ANALYZER
# ========================
class SmartMoneyAnalyzer:
    """Smart Money Concepts Analysis"""
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate S/R levels"""
        if len(df) < 20:
            return None, None
        
        last_20 = df.tail(20)
        support = last_20['low'].min()
        resistance = last_20['high'].max()
        
        logger.info(f"📊 S/R: Support=₹{support:.2f}, Resistance=₹{resistance:.2f}")
        return support, resistance
    
    @staticmethod
    def analyze_oi(option_chain: Dict) -> Dict:
        """Analyze Open Interest with Redis comparison"""
        try:
            oc_data = option_chain.get('oc', {})
            
            if not oc_data:
                return {'pcr': 0, 'signal': 'NEUTRAL', 'confidence': 0}
            
            total_call_oi = 0
            total_put_oi = 0
            
            for strike_data in oc_data.values():
                total_call_oi += strike_data.get('ce', {}).get('oi', 0)
                total_put_oi += strike_data.get('pe', {}).get('oi', 0)
            
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Get Redis comparison
            comparison = option_chain.get('_redis_comparison', {})
            delta = comparison.get('delta', {})
            
            # Adjust signal based on OI changes
            if pcr > 1.5:
                signal, confidence = "BULLISH", 75
            elif pcr < 0.5:
                signal, confidence = "BEARISH", 75
            else:
                signal, confidence = "NEUTRAL", 50
            
            # Boost confidence if OI change supports signal
            if delta:
                put_oi_change = delta.get('put_oi_change', 0)
                call_oi_change = delta.get('call_oi_change', 0)
                
                if signal == "BULLISH" and put_oi_change > call_oi_change:
                    confidence += 10
                elif signal == "BEARISH" and call_oi_change > put_oi_change:
                    confidence += 10
            
            result = {
                'pcr': pcr,
                'signal': signal,
                'confidence': min(confidence, 95),
                'call_oi': total_call_oi,
                'put_oi': total_put_oi,
                'oi_comparison': delta
            }
            
            logger.info(f"📈 OI: PCR={pcr:.2f}, Signal={signal}, Confidence={confidence}%")
            return result
        
        except Exception as e:
            logger.error(f"❌ OI analysis error: {e}")
            return {'pcr': 0, 'signal': 'NEUTRAL', 'confidence': 0}
    
    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame) -> float:
        """Calculate volume ratio"""
        if len(df) < 20:
            return 1.0
        
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        logger.info(f"📊 Volume Ratio: {ratio:.2f}x")
        return ratio


# ========================
# DEEPSEEK AI ANALYZER
# ========================
class DeepSeekAnalyzer:
    """DeepSeek V3 AI Integration"""
    
    @staticmethod
    def analyze(context: Dict) -> Optional[Dict]:
        """Get AI analysis with OI comparison context"""
        try:
            logger.info("🤖 Calling DeepSeek AI...")
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Build OI comparison text
            oi_comp = context.get('oi_comparison', {})
            oi_text = ""
            if oi_comp and oi_comp.get('call_oi_change'):
                oi_text = f"""
OI Changes (last scan):
- Call OI: {oi_comp.get('call_oi_change', 0):+,.0f}
- Put OI: {oi_comp.get('put_oi_change', 0):+,.0f}
- PCR: {oi_comp.get('pcr_old', 0):.2f} → {oi_comp.get('pcr_new', 0):.2f} ({oi_comp.get('pcr_change', 0):+.2f})
"""
            
            prompt = f"""You are expert F&O trader. Analyze data and give signal in JSON.

DATA:
- Symbol: {context['symbol']}
- Spot Price: ₹{context['spot_price']}
- Support: ₹{context.get('support', 'N/A')}
- Resistance: ₹{context.get('resistance', 'N/A')}
- Patterns: {context['patterns']}
- PCR: {context['pcr']}
- OI Signal: {context['oi_signal']}
- Volume Ratio: {context['volume_ratio']}x
{oi_text}

Reply ONLY in JSON:
{{
  "signal": "BUY/SELL/WAIT",
  "confidence": 75,
  "entry": {context['spot_price']},
  "target": {context['spot_price'] * 1.02},
  "stop_loss": {context['spot_price'] * 0.98},
  "risk_reward": "1:2",
  "marathi_explanation": "Technical analysis in Marathi..."
}}
"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are expert trader. Reply in JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"❌ DeepSeek API error: {response.status_code}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                logger.info(f"✅ AI: {analysis['signal']} ({analysis['confidence']}%)")
                return analysis
            
            logger.warning("⚠️ Could not parse AI response")
            return None
        
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            logger.error(traceback.format_exc())
            return None


# ========================
# MAIN BOT
# ========================
class SmartMoneyBot:
    """Main Trading Bot"""
    
    def __init__(self):
        logger.info("🔧 Initializing SmartMoneyBot...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.running = True
        logger.info("✅ SmartMoneyBot initialized")
    
    def is_market_open(self) -> bool:
        """Check if market is open (IST timezone)"""
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            logger.info(f"📅 Weekend: Market closed (IST: {current_time})")
            return False
        
        if Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE:
            logger.info(f"✅ Market OPEN (IST: {current_time})")
            return True
        
        logger.info(f"⏰ Market closed (IST: {current_time})")
        return False
    
    async def scan_symbol(self, symbol: str, info: Dict):
        """Scan single symbol"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 SCANNING: {symbol}")
            logger.info(f"{'='*60}")
            
            # Get expiry
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No expiry - SKIP")
                return
            
            # Get candles
            candles_df = self.dhan.get_historical_candles(security_id, segment, symbol)
            if candles_df is None or len(candles_df) < 20:
                logger.warning(f"⚠️ {symbol}: Insufficient candles - SKIP")
                return
            
            # Get option chain (with Redis caching)
            option_chain = self.dhan.get_option_chain(security_id, segment, expiry, symbol)
            if not option_chain:
                logger.warning(f"⚠️ {symbol}: No option chain - SKIP")
                return
            
            spot_price = option_chain.get('last_price', 0)
            logger.info(f"💰 Spot Price: ₹{spot_price}")
            
            # Technical Analysis
            logger.info(f"📊 Running technical analysis...")
            patterns = PatternDetector.detect_patterns(candles_df)
            support, resistance = SmartMoneyAnalyzer.calculate_support_resistance(candles_df)
            oi_analysis = SmartMoneyAnalyzer.analyze_oi(option_chain)
            volume_ratio = SmartMoneyAnalyzer.calculate_volume_ratio(candles_df)
            
            confluence = len(patterns) + 4
            if support and resistance:
                confluence += 1
            if volume_ratio > 1.5:
                confluence += 1
            if oi_analysis['confidence'] > 70:
                confluence += 1
            confluence = min(confluence, 10)
            
            logger.info(f"🎯 Confluence Score: {confluence}/10")
            
            # AI Analysis
            context = {
                'symbol': symbol,
                'spot_price': spot_price,
                'support': support,
                'resistance': resistance,
                'patterns': [p['name'] for p in patterns],
                'pcr': oi_analysis['pcr'],
                'oi_signal': oi_analysis['signal'],
                'volume_ratio': round(volume_ratio, 2),
                'confluence_score': confluence,
                'oi_comparison': oi_analysis.get('oi_comparison', {})
            }
            
            analysis = DeepSeekAnalyzer.analyze(context)
            
            if not analysis:
                logger.warning(f"⚠️ {symbol}: No AI analysis - SKIP")
                return
            
            # Check confidence
            if analysis['confidence'] < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"⏸️ {symbol}: Low confidence ({analysis['confidence']}%) - NO ALERT")
                return
            
            # Prepare OI comparison text for Telegram
            oi_comp = oi_analysis.get('oi_comparison', {})
            oi_change_text = ""
            if oi_comp and oi_comp.get('call_oi_change'):
                oi_change_text = f"""
📊 <b>OI Changes (vs last scan):</b>
• Call OI: {oi_comp.get('call_oi_change', 0):+,.0f}
• Put OI: {oi_comp.get('put_oi_change', 0):+,.0f}
• PCR: {oi_comp.get('pcr_old', 0):.2f} → {oi_comp.get('pcr_new', 0):.2f} ({oi_comp.get('pcr_change', 0):+.2f})
• Time: {oi_comp.get('time_diff', 0):.0f} mins ago
"""
            
            # Send Telegram alert
            signal_emoji = "🟢" if analysis['signal'] == "BUY" else "🔴" if analysis['signal'] == "SELL" else "⚪"
            
            message = f"""
🚀 <b>SMART MONEY SIGNAL</b>

📊 Symbol: <b>{symbol}</b>
💰 Spot: ₹{spot_price:,.2f}
📅 Expiry: {expiry}
⏰ Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{signal_emoji} Signal: <b>{analysis['signal']}</b>
💪 Confidence: <b>{analysis['confidence']}%</b>

🎯 <b>TRADE SETUP:</b>
• Entry: ₹{analysis.get('entry', spot_price):,.2f}
• Target: ₹{analysis.get('target', spot_price * 1.02):,.2f}
• Stop-Loss: ₹{analysis.get('stop_loss', spot_price * 0.98):,.2f}
• Risk/Reward: {analysis.get('risk_reward', '1:2')}

📈 <b>ANALYSIS:</b>
• Support: ₹{support:,.0f if support else 'N/A'}
• Resistance: ₹{resistance:,.0f if resistance else 'N/A'}
• PCR: {oi_analysis['pcr']:.2f}
• Volume: {volume_ratio:.2f}x avg
• Patterns: {len(patterns)} detected
• Confluence: {confluence}/10
{oi_change_text}
📝 <b>Marathi Explanation:</b>
{analysis.get('marathi_explanation', 'Technical analysis complete.')}

⚡ Disclaimer: Trade at your own risk.
💾 Data cached in Redis for comparison
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info(f"✅ {symbol}: ALERT SENT TO TELEGRAM! 🎉")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_startup_message(self):
        """Send startup notification"""
        try:
            logger.info("📤 Sending startup message to Telegram...")
            ist = pytz.timezone('Asia/Kolkata')
            
            redis_status = "✅ Connected" if self.redis.redis_client else "❌ Disconnected"
            
            msg = f"""
🤖 <b>Smart Money Bot v3.0 Started!</b>

📊 Tracking: <b>{len(self.dhan.security_id_map)} symbols</b>
⏰ Scan Interval: {Config.SCAN_INTERVAL//60} minutes
🎯 Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}%
⏱️ Market Hours: {Config.MARKET_OPEN} - {Config.MARKET_CLOSE} IST
🔴 Redis Cache: {redis_status}

🔍 <b>Features:</b>
✅ Auto nearest expiry selection
✅ Historical 5-min candles analysis
✅ Smart Money Concepts (OI, PCR, Volume)
✅ Candlestick pattern detection
✅ DeepSeek AI reasoning
✅ Redis caching for OI comparison
✅ Real-time data delta tracking

📈 <b>Active Symbols ({len(self.dhan.security_id_map)}):</b>
{', '.join(self.dhan.security_id_map.keys())}

⚡ Powered by: DhanHQ API + DeepSeek AI + Redis
🚀 Status: <b>ACTIVE & READY</b> ✅
⏰ Startup: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}

📝 Next scan starts when market opens!
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("✅ Startup message sent successfully!")
        except Exception as e:
            logger.error(f"❌ Startup message error: {e}")
            logger.error(traceback.format_exc())
    
    async def run(self):
        """Main bot loop"""
        logger.info("="*60)
        logger.info("🚀 SMART MONEY BOT v3.0 STARTING...")
        logger.info("="*60)
        
        # Validate credentials
        logger.info("🔐 Validating API credentials...")
        missing = []
        if not Config.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not Config.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        if not Config.DHAN_CLIENT_ID:
            missing.append("DHAN_CLIENT_ID")
        if not Config.DHAN_ACCESS_TOKEN:
            missing.append("DHAN_ACCESS_TOKEN")
        if not Config.DEEPSEEK_API_KEY:
            missing.append("DEEPSEEK_API_KEY")
        
        if missing:
            logger.error(f"❌ Missing credentials: {', '.join(missing)}")
            return
        
        logger.info("✅ All API credentials validated!")
        
        # Load security IDs
        logger.info("📥 Loading security IDs from Dhan...")
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("❌ Failed to load securities. Exiting...")
            return
        
        logger.info(f"✅ Loaded {len(self.dhan.security_id_map)} securities!")
        
        # Send startup message
        await self.send_startup_message()
        
        logger.info("="*60)
        logger.info("🎯 Bot is now RUNNING! Monitoring market...")
        logger.info("="*60)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("😴 Market closed. Sleeping for 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 SCAN CYCLE START")
                logger.info(f"⏰ IST Time: {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"📊 Scanning {len(self.dhan.security_id_map)} symbols...")
                logger.info(f"{'='*60}")
                
                # Scan each symbol
                for idx, (symbol, info) in enumerate(self.dhan.security_id_map.items(), 1):
                    logger.info(f"\n[{idx}/{len(self.dhan.security_id_map)}] Processing {symbol}...")
                    await self.scan_symbol(symbol, info)
                    await asyncio.sleep(3)  # Rate limit between symbols
                
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ SCAN CYCLE COMPLETE!")
                logger.info(f"⏰ Next scan in {Config.SCAN_INTERVAL//60} minutes...")
                logger.info(f"{'='*60}\n")
                
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
        logger.info("="*60)
        logger.info("🚀 INITIALIZING SMART MONEY BOT v3.0")
        logger.info("="*60)
        
        bot = SmartMoneyBot()
        await bot.run()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("="*60)
        logger.info("👋 Bot shutdown complete")
        logger.info("="*60)


if __name__ == "__main__":
    ist = pytz.timezone('Asia/Kolkata')
    logger.info("="*60)
    logger.info("🎬 BOT STARTING...")
    logger.info(f"⏰ IST: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🌍 Timezone: Asia/Kolkata (IST)")
    logger.info("="*60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        logger.error(traceback.format_exc())
