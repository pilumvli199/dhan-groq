import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import json
import redis
from collections import defaultdict
import numpy as np
import pytz

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Indian timezone
IST = pytz.timezone('Asia/Kolkata')

# ========================
# CONFIGURATION
# ========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# PRIORITY STOCKS LIST - NSE_EQ Security IDs (for option chain API)
STOCKS_INDICES = {
    # Indices (Must Track)
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I", "security_id": 13},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I", "security_id": 25},
    "FINNIFTY": {"symbol": "FINNIFTY", "segment": "IDX_I", "security_id": 27},
    
    # 🔥 FIX: Use NSE_EQ security IDs for option chain API!
    # These IDs are from NSE_EQ segment, but API accepts them for FNO option chains
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ", "security_id": 2885},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ", "security_id": 1333},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ", "security_id": 1594},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ", "security_id": 1330},
    "TCS": {"symbol": "TCS", "segment": "NSE_EQ", "security_id": 11536},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ", "security_id": 3045},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ", "security_id": 392},
    "ITC": {"symbol": "ITC", "segment": "NSE_EQ", "security_id": 1660},
    "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ", "security_id": 1922},
    "LT": {"symbol": "LT", "segment": "NSE_EQ", "security_id": 2672},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ", "security_id": 5900},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ", "security_id": 317},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ", "security_id": 3456},
    "TATASTEEL": {"symbol": "TATASTEEL", "segment": "NSE_EQ", "security_id": 3499},
    "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ", "security_id": 10999},
}

# ========================
# BOT CLASS
# ========================
class SmartTradingBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.running = True
        self.headers = {
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID,
            'Content-Type': 'application/json'
        }
        
        # Redis connection
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connected!")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self.redis_client = None
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Failed symbols tracking
        self.failed_symbols = set()
        
        # 🔥 Rate limiting: Option chain API = 1 request per 3 seconds
        self.last_option_chain_call = 0
        self.option_chain_rate_limit = 3  # seconds
        
        logger.info("Bot initialized successfully")
    
    def test_api_connection(self):
        """Test Dhan API connection AND fetch correct security IDs"""
        logger.info("\n" + "="*80)
        logger.info("🧪 TESTING DHAN API & FETCHING INSTRUMENT LIST")
        logger.info("="*80)
        
        # Step 1: Download NSE_FO instrument list
        logger.info("\n📥 Downloading NSE_FO Instrument List...")
        try:
            instrument_url = "https://api.dhan.co/v2/instrument/NSE_FO"
            response = self.session.get(instrument_url, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Downloaded! Size: {len(response.text)} bytes")
                
                # Parse CSV
                import csv
                import io
                
                csv_data = io.StringIO(response.text)
                reader = csv.DictReader(csv_data)
                
                # Find our stocks
                stock_symbols = ['RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'TCS', 
                               'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT',
                               'AXISBANK', 'BAJFINANCE', 'TATAMOTORS', 'TATASTEEL', 'MARUTI']
                
                found_stocks = {}
                
                for row in reader:
                    symbol = row.get('SEM_TRADING_SYMBOL', '').replace('-EQ', '')
                    if symbol in stock_symbols:
                        sec_id = row.get('SEM_SMST_SECURITY_ID')
                        expiry = row.get('SEM_EXPIRY_DATE')
                        instrument = row.get('SEM_INSTRUMENT_NAME')
                        
                        if instrument and 'FUT' in instrument:  # Only futures
                            if symbol not in found_stocks:
                                found_stocks[symbol] = {
                                    'security_id': sec_id,
                                    'expiry': expiry,
                                    'instrument': instrument
                                }
                
                logger.info(f"\n📊 Found {len(found_stocks)} FNO stocks:")
                for symbol, data in found_stocks.items():
                    logger.info(f"  {symbol}: SecurityID={data['security_id']}, Expiry={data['expiry']}")
                
                if not found_stocks:
                    logger.warning("⚠️ No FNO stocks found! Need to use NSE_EQ security IDs for option chain")
                    
            else:
                logger.error(f"❌ Failed to download: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Step 2: Test with indices (working)
        logger.info("\n📊 Test: NIFTY 50 (Known Working)")
        try:
            payload = {
                "UnderlyingScrip": 13,
                "UnderlyingSeg": "IDX_I"
            }
            logger.info(f"Request: {payload}")
            
            response = self.session.post(
                DHAN_EXPIRY_LIST_URL,
                json=payload,
                timeout=10
            )
            
            logger.info(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    logger.info(f"✅ SUCCESS! Found {len(data.get('data', []))} expiries")
                    logger.info(f"   Expiries: {data.get('data', [])[:3]}")
            else:
                logger.info(f"Response: {response.text}")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        
        logger.info("\n" + "="*80)
        logger.info("💡 SOLUTION: Use NSE_EQ security IDs for stocks!")
        logger.info("="*80 + "\n")
    
    def get_ist_time(self):
        """Current Indian time"""
        return datetime.now(IST)
    
    def format_ist_time(self, dt=None):
        """IST time format"""
        if dt is None:
            dt = self.get_ist_time()
        return dt.strftime('%Y-%m-%d %H:%M:%S IST')
    
    def is_market_hours(self):
        """Check if market is open"""
        now = self.get_ist_time()
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:15 AM - 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        
        return market_open <= now <= market_close
    
    def get_historical_candles(self, security_id, segment, symbol):
        """Last 50 x 5-minute candles"""
        try:
            # FIX: Always use NSE_EQ for stocks' candle data
            exch_seg = "IDX_I" if segment == "IDX_I" else "NSE_EQ"
            instrument = "INDEX" if segment == "IDX_I" else "EQUITY"
            
            to_date = self.get_ist_time()
            from_date = to_date - timedelta(days=7)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exch_seg,
                "instrument": instrument,
                "interval": "5",
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            response = self.session.post(
                DHAN_INTRADAY_URL,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different timestamp formats
                timestamp_field = None
                for field in ['start_Time', 'timestamp', 'time', 'start_time']:
                    if field in data:
                        timestamp_field = field
                        break
                
                if not timestamp_field:
                    timestamps = [f"{i:04d}" for i in range(len(data.get('open', [])))]
                else:
                    timestamps = data[timestamp_field]
                
                opens = data.get('open', [])
                highs = data.get('high', [])
                lows = data.get('low', [])
                closes = data.get('close', [])
                volumes = data.get('volume', [])
                
                if not opens:
                    return None
                
                candles = []
                length = min(len(opens), len(timestamps))
                
                for i in range(length):
                    try:
                        candles.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else f"{i:04d}",
                            'open': float(opens[i]) if i < len(opens) else 0,
                            'high': float(highs[i]) if i < len(highs) else 0,
                            'low': float(lows[i]) if i < len(lows) else 0,
                            'close': float(closes[i]) if i < len(closes) else 0,
                            'volume': int(volumes[i]) if i < len(volumes) else 0
                        })
                    except (ValueError, TypeError):
                        continue
                
                if not candles:
                    return None
                
                return candles[-50:] if len(candles) > 50 else candles
            
            return None
            
        except Exception as e:
            logger.debug(f"{symbol}: Candles error: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Option chain data - FIXED: Use NSE_EQ for stocks"""
        try:
            # 🔥 RATE LIMITING: Wait if needed (1 request per 3 seconds)
            import time
            current_time = time.time()
            time_since_last_call = current_time - self.last_option_chain_call
            
            if time_since_last_call < self.option_chain_rate_limit:
                sleep_time = self.option_chain_rate_limit - time_since_last_call
                logger.debug(f"  ⏱️ Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            # 🔥 CRITICAL FIX: Always use NSE_EQ for stocks' option chains!
            api_segment = "IDX_I" if segment == "IDX_I" else "NSE_EQ"
            
            payload = {
                "UnderlyingScrip": int(security_id),  # Must be integer
                "UnderlyingSeg": api_segment,
                "Expiry": expiry
            }
            
            logger.debug(f"  Option Chain Request: {payload}")
            
            response = self.session.post(
                DHAN_OPTION_CHAIN_URL,
                json=payload,
                timeout=15
            )
            
            # Update last call time
            self.last_option_chain_call = time.time()
            
            logger.debug(f"  Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Per documentation: {"data": {"last_price": ..., "oc": {...}}, "status": "success"}
                if isinstance(result, dict):
                    if result.get('status') == 'success' and result.get('data'):
                        return result['data']
                    
                    # Alternative format (direct data)
                    if result.get('oc') and result.get('last_price'):
                        return result
                    
                    # Error case
                    if result.get('status') == 'failure':
                        logger.debug(f"  API Error: {result.get('remarks', 'Unknown')}")
                        return None
                
                logger.debug(f"  Unexpected response format")
                return None
            else:
                logger.debug(f"  HTTP Error: {response.status_code}")
                return None
            
        except Exception as e:
            logger.debug(f"  Exception: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """Get nearest expiry - WORKING FORMAT from your other bot"""
        try:
            # 🔥 WORKING FORMAT: Always use NSE_EQ for stocks
            api_segment = "IDX_I" if segment == "IDX_I" else "NSE_EQ"
            
            payload = {
                "UnderlyingScrip": int(security_id),  # Integer
                "UnderlyingSeg": api_segment
            }
            
            logger.info(f"  📤 Expiry Request: {payload}")
            
            response = self.session.post(
                DHAN_EXPIRY_LIST_URL,
                json=payload,
                timeout=10
            )
            
            logger.info(f"  📥 Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Response format: {"data": ["2025-10-20", ...], "status": "success"}
                if data.get('status') == 'success' and data.get('data'):
                    expiries = data['data']
                    if expiries:
                        logger.info(f"  ✅ Found {len(expiries)} expiries")
                        logger.info(f"  ✅ Nearest: {expiries[0]}")
                        return expiries[0]
                
                logger.warning(f"  ⚠️ No expiry data: {data}")
                return None
            else:
                logger.error(f"  ❌ HTTP {response.status_code}: {response.text[:200]}")
                return None
            
        except Exception as e:
            logger.error(f"  ❌ Exception: {e}")
            return None
    
    def filter_atm_strikes(self, oc_data, spot_price):
        """Filter ATM ±5 strikes"""
        try:
            all_strikes = oc_data.get('oc', {})
            if not all_strikes:
                return {}
            
            strike_map = {}
            strike_prices = []
            
            for strike_key in all_strikes.keys():
                try:
                    strike_float = float(strike_key)
                    strike_prices.append(strike_float)
                    strike_map[strike_float] = strike_key
                except:
                    continue
            
            if not strike_prices:
                return {}
            
            strike_prices.sort()
            atm_strike = min(strike_prices, key=lambda x: abs(x - spot_price))
            atm_index = strike_prices.index(atm_strike)
            
            start_idx = max(0, atm_index - 5)
            end_idx = min(len(strike_prices), atm_index + 6)
            
            filtered_strikes = strike_prices[start_idx:end_idx]
            
            filtered_oc = {
                'last_price': oc_data.get('last_price'),
                'oc': {}
            }
            
            for strike_float in filtered_strikes:
                original_key = strike_map.get(strike_float)
                if original_key and original_key in all_strikes:
                    filtered_oc['oc'][original_key] = all_strikes[original_key]
            
            logger.info(f"  ├─ ATM Strike: ₹{atm_strike:.0f}")
            logger.info(f"  ├─ Filtered: {len(filtered_oc['oc'])} strikes")
            if filtered_strikes:
                logger.info(f"  └─ Range: ₹{min(filtered_strikes):.0f} to ₹{max(filtered_strikes):.0f}")
            
            return filtered_oc
            
        except Exception as e:
            logger.error(f"  Error filtering strikes: {e}")
            return oc_data
    
    def calculate_oi_changes(self, current_oc, symbol):
        """Calculate OI changes"""
        try:
            if not self.redis_client:
                return {}
            
            prev_key = f"oi:current:{symbol}"
            prev_data = self.redis_client.get(prev_key)
            
            if not prev_data:
                self.redis_client.setex(prev_key, 3600, json.dumps(current_oc))
                return {}
            
            prev_oc = json.loads(prev_data)
            
            oi_changes = {}
            current_strikes = current_oc.get('oc', {})
            prev_strikes = prev_oc.get('oc', {})
            
            for strike, data in current_strikes.items():
                ce_current = data.get('ce', {}).get('oi', 0)
                pe_current = data.get('pe', {}).get('oi', 0)
                
                ce_prev = prev_strikes.get(strike, {}).get('ce', {}).get('oi', 0)
                pe_prev = prev_strikes.get(strike, {}).get('pe', {}).get('oi', 0)
                
                oi_changes[strike] = {
                    'ce_oi_change': ce_current - ce_prev,
                    'pe_oi_change': pe_current - pe_prev,
                    'ce_volume': data.get('ce', {}).get('volume', 0),
                    'pe_volume': data.get('pe', {}).get('volume', 0)
                }
            
            self.redis_client.setex(prev_key, 3600, json.dumps(current_oc))
            
            return oi_changes
            
        except Exception as e:
            logger.error(f"Error calculating OI: {e}")
            return {}
    
    def detect_candlestick_pattern(self, candles):
        """Detect candlestick patterns"""
        if len(candles) < 3:
            return None
        
        c1, c2 = candles[-2:]
        
        # Bullish Engulfing
        if (c1['close'] < c1['open'] and
            c2['close'] > c2['open'] and
            c2['open'] < c1['close'] and
            c2['close'] > c1['open']):
            return 'bullish_engulfing'
        
        # Bearish Engulfing
        if (c1['close'] > c1['open'] and
            c2['close'] < c2['open'] and
            c2['open'] > c1['close'] and
            c2['close'] < c1['open']):
            return 'bearish_engulfing'
        
        # Hammer
        body = abs(c2['close'] - c2['open'])
        lower_shadow = min(c2['open'], c2['close']) - c2['low']
        upper_shadow = c2['high'] - max(c2['open'], c2['close'])
        
        if body > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.3:
            return 'hammer'
        
        # Shooting Star
        if body > 0 and upper_shadow > body * 2 and lower_shadow < body * 0.3:
            return 'shooting_star'
        
        return None
    
    def calculate_ema(self, candles, period):
        """Calculate EMA"""
        closes = [c['close'] for c in candles]
        
        if len(closes) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = closes[0]
        
        for price in closes[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_support_resistance(self, candles):
        """Calculate Support/Resistance"""
        if len(candles) < 20:
            return None, None
        
        highs = [c['high'] for c in candles[-20:]]
        lows = [c['low'] for c in candles[-20:]]
        
        return min(lows), max(highs)
    
    def calculate_signal_score(self, oi_changes, candles, spot_price, oc_data):
        """Bot's scoring algorithm"""
        try:
            score = 0
            signal_type = None
            details = {}
            
            # OI ANALYSIS (50 points)
            oi_score = 0
            
            total_ce_oi = sum(data.get('ce', {}).get('oi', 0) 
                            for data in oc_data.get('oc', {}).values())
            total_pe_oi = sum(data.get('pe', {}).get('oi', 0) 
                            for data in oc_data.get('oc', {}).values())
            
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
            details['pcr'] = round(pcr, 2)
            
            if pcr < 0.8:
                oi_score += 15
                signal_type = "BEARISH"
            elif pcr > 1.2:
                oi_score += 15
                signal_type = "BULLISH"
            elif pcr < 0.9 or pcr > 1.1:
                oi_score += 8
            
            major_changes = []
            for strike, change in oi_changes.items():
                ce_change = change['ce_oi_change']
                pe_change = change['pe_oi_change']
                
                if abs(ce_change) > 100000:
                    oi_score += 5
                    major_changes.append({
                        'strike': strike,
                        'type': 'CE',
                        'change': ce_change
                    })
                
                if abs(pe_change) > 100000:
                    oi_score += 5
                    major_changes.append({
                        'strike': strike,
                        'type': 'PE',
                        'change': pe_change
                    })
            
            oi_score = min(oi_score, 50)
            details['major_oi_changes'] = major_changes[:5]
            
            # CHART ANALYSIS (50 points)
            chart_score = 0
            
            pattern = self.detect_candlestick_pattern(candles)
            details['pattern'] = pattern
            
            if pattern in ['bullish_engulfing', 'hammer']:
                chart_score += 20
                if not signal_type:
                    signal_type = "BULLISH"
            elif pattern in ['bearish_engulfing', 'shooting_star']:
                chart_score += 20
                if not signal_type:
                    signal_type = "BEARISH"
            
            if len(candles) >= 10:
                last_volume = candles[-1]['volume']
                avg_volume = sum(c['volume'] for c in candles[-10:-1]) / 9
                
                if avg_volume > 0:
                    if last_volume > avg_volume * 1.5:
                        chart_score += 10
                    elif last_volume > avg_volume * 1.2:
                        chart_score += 5
            
            support, resistance = self.calculate_support_resistance(candles)
            details['support'] = round(support, 2) if support else 0
            details['resistance'] = round(resistance, 2) if resistance else 0
            
            if resistance and spot_price > resistance:
                chart_score += 10
                if not signal_type:
                    signal_type = "BULLISH"
            elif support and spot_price < support:
                chart_score += 10
                if not signal_type:
                    signal_type = "BEARISH"
            
            ema9 = self.calculate_ema(candles, 9)
            ema21 = self.calculate_ema(candles, 21)
            
            if ema9 and ema21:
                if ema9 > ema21 and signal_type == "BULLISH":
                    chart_score += 10
                elif ema9 < ema21 and signal_type == "BEARISH":
                    chart_score += 10
            
            chart_score = min(chart_score, 50)
            
            total_score = oi_score + chart_score
            
            return {
                'total_score': total_score,
                'oi_score': oi_score,
                'chart_score': chart_score,
                'signal_type': signal_type or "NEUTRAL",
                'confidence': 'HIGH' if total_score >= 70 else 'MEDIUM' if total_score >= 50 else 'LOW',
                'send_to_ai': total_score >= 70,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return None
    
    def prepare_ai_data(self, symbol, spot, oi_changes, candles, bot_analysis):
        """Prepare data for AI"""
        oi_summary = {
            'pcr': bot_analysis['details'].get('pcr', 1.0),
            'significant_changes': []
        }
        
        for change in bot_analysis['details'].get('major_oi_changes', [])[:5]:
            oi_summary['significant_changes'].append({
                'strike': int(float(change['strike'])),
                'type': change['type'],
                'change_k': int(change['change'] / 1000)
            })
        
        candles_data = []
        for candle in candles[-20:]:
            candles_data.append({
                'time': candle['timestamp'][-8:] if len(candle['timestamp']) >= 8 else candle['timestamp'],
                'o': round(candle['open'], 1),
                'h': round(candle['high'], 1),
                'l': round(candle['low'], 1),
                'c': round(candle['close'], 1),
                'v': int(candle['volume'] / 1000)
            })
        
        return {
            'symbol': symbol,
            'timestamp': self.format_ist_time(),
            'spot_price': round(spot, 2),
            'bot_analysis': {
                'signal': bot_analysis['signal_type'],
                'score': bot_analysis['total_score'],
                'oi_score': bot_analysis['oi_score'],
                'chart_score': bot_analysis['chart_score']
            },
            'oi_data': oi_summary,
            'candles': candles_data,
            'pattern': bot_analysis['details'].get('pattern', 'None'),
            'support': bot_analysis['details'].get('support', 0),
            'resistance': bot_analysis['details'].get('resistance', 0)
        }
    
    async def verify_with_ai(self, ai_data):
        """DeepSeek AI verification"""
        try:
            prompt = f"""Expert options trader. Verify signal.

BOT: {ai_data['bot_analysis']['signal']} ({ai_data['bot_analysis']['score']}/100)
Symbol: {ai_data['symbol']} | Spot: ₹{ai_data['spot_price']}
PCR: {ai_data['oi_data']['pcr']} | Pattern: {ai_data['pattern']}

JSON only:
{{"status":"CONFIRMED/REJECTED","entry":25700,"target":25800,"stop_loss":25650,"risk_reward":1.5,"confidence":80,"reasoning":["Point 1","Point 2"]}}"""

            headers_ai = {
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.3,
                'max_tokens': 300
            }
            
            response = requests.post(
                DEEPSEEK_API_URL,
                json=payload,
                headers=headers_ai,
                timeout=15
            )
            
            if response.status_code == 200:
                ai_response = response.json()
                content = ai_response['choices'][0]['message']['content']
                
                import re
                json_match = re.search(r'\{[^{}]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
                
                return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.debug(f"AI error: {e}")
            return None
    
    async def send_signal(self, symbol, ai_data, ai_response, bot_analysis):
        """Send signal to Telegram"""
        try:
            if ai_response['status'] == 'REJECTED':
                msg = f"❌ REJECTED - {symbol}\nScore: {bot_analysis['total_score']}/100\n{ai_response['reasoning'][0]}"
                await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                return
            
            signal_emoji = "🟢" if bot_analysis['signal_type'] == 'BULLISH' else "🔴"
            
            msg = f"""{signal_emoji} TRADE SIGNAL {signal_emoji}

━━━━━━━━━━━━━━━━━━━━
📊 {symbol}
━━━━━━━━━━━━━━━━━━━━

⏰ {ai_data['timestamp']}
💰 Spot: ₹{ai_data['spot_price']:,.2f}
Direction: {bot_analysis['signal_type']}

━━━━━━━━━━━━━━━━━━━━
🎯 SETUP
━━━━━━━━━━━━━━━━━━━━

Entry: ₹{ai_response['entry']:,.0f}
Target: ₹{ai_response['target']:,.0f}
SL: ₹{ai_response['stop_loss']:,.0f}
R:R: 1:{ai_response['risk_reward']:.1f}

━━━━━━━━━━━━━━━━━━━━
SCORES
━━━━━━━━━━━━━━━━━━━━

Bot: {bot_analysis['total_score']}/100
AI: {ai_response['confidence']}%
PCR: {ai_data['oi_data']['pcr']}

━━━━━━━━━━━━━━━━━━━━
"""
            
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
    
    async def analyze_symbol(self, symbol, info):
        """Analyze single symbol"""
        try:
            # Skip if previously failed multiple times
            if symbol in self.failed_symbols:
                return
            
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"🔍 {symbol} | {self.format_ist_time()}")
            logger.info(f"{'─'*60}")
            
            # Step 1: Get expiry
            expiry = self.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"  ❌ {symbol}: No expiry")
                self.failed_symbols.add(symbol)
                return
            
            logger.info(f"  ✅ Expiry: {expiry}")
            
            # Step 2: Get option chain
            oc_data = self.get_option_chain(security_id, segment, expiry)
            if not oc_data:
                logger.warning(f"  ❌ {symbol}: No option chain")
                return
            
            spot_price = oc_data.get('last_price', 0)
            if not spot_price:
                logger.warning(f"  ❌ {symbol}: No spot price")
                return
            
            logger.info(f"  ✅ Spot: ₹{spot_price:,.2f}")
            logger.info(f"  ✅ Strikes: {len(oc_data.get('oc', {}))}")
            
            # Step 3: Filter ATM strikes
            filtered_oc = self.filter_atm_strikes(oc_data, spot_price)
            if not filtered_oc or not filtered_oc.get('oc'):
                logger.warning(f"  ❌ {symbol}: No strikes after filtering")
                return
            
            # Step 4: Get candles
            candles = self.get_historical_candles(security_id, segment, symbol)
            if not candles or len(candles) < 20:
                logger.warning(f"  ❌ {symbol}: Not enough candles")
                return
            
            logger.info(f"  ✅ Candles: {len(candles)} × 5min")
            
            # Step 5: Calculate OI changes
            oi_changes = self.calculate_oi_changes(filtered_oc, symbol)
            if not oi_changes:
                logger.info(f"  ℹ️ {symbol}: First run - baseline saved")
                return
            
            total_ce_change = sum(c.get('ce_oi_change', 0) for c in oi_changes.values())
            total_pe_change = sum(c.get('pe_oi_change', 0) for c in oi_changes.values())
            logger.info(f"  ✅ OI: CE{total_ce_change:+,.0f} PE{total_pe_change:+,.0f}")
            
            # Step 6: Bot analysis
            logger.info(f"  🤖 Analyzing...")
            bot_analysis = self.calculate_signal_score(
                oi_changes, candles, spot_price, filtered_oc
            )
            
            if not bot_analysis:
                logger.warning(f"  ❌ {symbol}: Analysis failed")
                return
            
            logger.info(f"  📊 Score: {bot_analysis['total_score']}/100")
            logger.info(f"    ├─ OI: {bot_analysis['oi_score']}/50")
            logger.info(f"    ├─ Chart: {bot_analysis['chart_score']}/50")
            logger.info(f"    ├─ Signal: {bot_analysis['signal_type']}")
            logger.info(f"    └─ PCR: {bot_analysis['details'].get('pcr', 0):.2f}")
            
            # Step 7: Check if high score
            if not bot_analysis['send_to_ai']:
                logger.info(f"  ⚠️ {symbol}: Score too low")
                return
            
            # HIGH SCORE ALERT!
            logger.info(f"\n{'='*60}")
            logger.info(f"🔥 HIGH SCORE: {symbol} - {bot_analysis['total_score']}/100")
            logger.info(f"{'='*60}")
            
            # Step 8: Prepare AI data
            ai_data = self.prepare_ai_data(
                symbol, spot_price, oi_changes, candles, bot_analysis
            )
            
            # Step 9: AI verification
            logger.info(f"🤖 Verifying with AI...")
            ai_response = await self.verify_with_ai(ai_data)
            
            if not ai_response:
                logger.warning(f"❌ AI verification failed")
                return
            
            logger.info(f"✅ AI: {ai_response['status']} ({ai_response.get('confidence', 0)}%)")
            
            # Step 10: Send signal
            await self.send_signal(symbol, ai_data, ai_response, bot_analysis)
            
        except Exception as e:
            logger.error(f"❌ {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    async def analyze_batch_parallel(self, batch):
        """Analyze batch in parallel"""
        tasks = []
        for symbol in batch:
            info = STOCKS_INDICES[symbol]
            tasks.append(self.analyze_symbol(symbol, info))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Smart Trading Bot Started!")
        
        # Check market hours
        if not self.is_market_hours():
            logger.warning("⚠️ Market is CLOSED!")
            logger.info("Market hours: Mon-Fri 9:15 AM - 3:30 PM IST")
            now = self.get_ist_time()
            logger.info(f"Current time: {self.format_ist_time(now)}")
            
            # Send Telegram notification
            try:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"⚠️ Bot started but market is CLOSED\n\nCurrent: {self.format_ist_time(now)}\nMarket: Mon-Fri 9:15 AM - 3:30 PM IST"
                )
            except:
                pass
        
        # Startup message
        await self.send_startup_message()
        
        # Symbol batches
        all_symbols = list(STOCKS_INDICES.keys())
        batch_size = 3  # 🔥 Reduced to 3 due to rate limiting (1 req per 3 sec)
        batches = [all_symbols[i:i+batch_size] 
                  for i in range(0, len(all_symbols), batch_size)]
        
        logger.info(f"📊 Tracking {len(all_symbols)} symbols in {len(batches)} batches")
        logger.info(f"⚡ Rate limit: 1 option chain request per 3 seconds")
        
        while self.running:
            try:
                # Check market hours
                if not self.is_market_hours():
                    logger.info("⏸️ Market closed - waiting...")
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                cycle_start = self.get_ist_time()
                logger.info(f"\n{'#'*80}")
                logger.info(f"CYCLE START: {self.format_ist_time(cycle_start)}")
                logger.info(f"{'#'*80}\n")
                
                # Process batches
                for batch_num, batch in enumerate(batches, 1):
                    batch_start = self.get_ist_time()
                    logger.info(f"\n📦 Batch {batch_num}/{len(batches)}: {batch}")
                    
                    await self.analyze_batch_parallel(batch)
                    
                    batch_duration = (self.get_ist_time() - batch_start).total_seconds()
                    logger.info(f"✅ Batch {batch_num} done in {batch_duration:.1f}s")
                    
                    # Wait between batches
                    if batch_num < len(batches):
                        logger.info(f"⏸️ Waiting 5s...")
                        await asyncio.sleep(5)
                
                cycle_end = self.get_ist_time()
                duration = (cycle_end - cycle_start).total_seconds()
                
                # Sleep calculation
                target_cycle = 300  # 5 minutes
                sleep_time = max(30, target_cycle - duration)
                
                logger.info(f"\n{'#'*80}")
                logger.info(f"CYCLE COMPLETE: {duration:.1f}s")
                logger.info(f"Sleeping: {sleep_time:.0f}s")
                next_time = self.get_ist_time() + timedelta(seconds=sleep_time)
                logger.info(f"Next: {next_time.strftime('%I:%M:%S %p IST')}")
                logger.info(f"{'#'*80}\n")
                
                await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Send startup notification"""
        try:
            market_status = "🟢 OPEN" if self.is_market_hours() else "🔴 CLOSED"
            
            msg = f"""🤖 SMART TRADING BOT STARTED!

━━━━━━━━━━━━━━━━━━━━
📊 STATUS
━━━━━━━━━━━━━━━━━━━━

Market: {market_status}
Time: {self.format_ist_time()}
Symbols: {len(STOCKS_INDICES)}
Update: Every 5 minutes

━━━━━━━━━━━━━━━━━━━━
✅ FEATURES
━━━━━━━━━━━━━━━━━━━━

✓ Bot Analysis (70+ score)
✓ AI Verification (DeepSeek)
✓ ATM ±5 Strikes Filter
✓ OI + Chart Analysis
✓ Auto Expiry Selection

━━━━━━━━━━━━━━━━━━━━
📈 TRACKING
━━━━━━━━━━━━━━━━━━━━

Indices:
  • NIFTY 50
  • NIFTY BANK
  • FINNIFTY

Top FNO Stocks:
  • RELIANCE, HDFCBANK
  • INFY, ICICIBANK, TCS
  • SBIN, BHARTIARTL, ITC
  • And 7 more...

━━━━━━━━━━━━━━━━━━━━
Ready to trade! 🎯
"""
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg
            )
            
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")


# ═══════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        # Environment check
        required_vars = {
            'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
            'DHAN_CLIENT_ID': DHAN_CLIENT_ID,
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN,
            'DEEPSEEK_API_KEY': DEEPSEEK_API_KEY
        }
        
        missing = [k for k, v in required_vars.items() if not v]
        
        if missing:
            logger.error(f"❌ Missing environment variables: {', '.join(missing)}")
            exit(1)
        
        logger.info("✅ All environment variables found!")
        
        bot = SmartTradingBot()
        
        # 🔥 RUN API TEST FIRST
        bot.test_api_connection()
        
        # Ask user to continue
        logger.info("\n" + "="*80)
        logger.info("⚠️  Check the API test results above")
        logger.info("Press CTRL+C to stop, or wait 10 seconds to continue...")
        logger.info("="*80 + "\n")
        
        import time
        time.sleep(10)
        
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
