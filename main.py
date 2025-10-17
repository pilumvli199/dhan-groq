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
import pytz  # For timezone handling

# Logging setup (INFO level for detailed analysis logs)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Shows all analysis steps
)
logger = logging.getLogger(__name__)

# Suppress httpx logs (too verbose)
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

# ALL STOCKS LIST - TOP 50 FNO STOCKS (Options-Enabled)
STOCKS_INDICES = {
    # Indices (Must Track - High Volume)
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I", "security_id": 13},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I", "security_id": 25},
    "FINNIFTY": {"symbol": "FINNIFTY", "segment": "IDX_I", "security_id": 27},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I", "security_id": 51},
    "MIDCPNIFTY": {"symbol": "MIDCPNIFTY", "segment": "IDX_I", "security_id": 288},
    
    # Top 45 FNO Stocks (High Volume + Options Active)
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ", "security_id": 2885},
    "TCS": {"symbol": "TCS", "segment": "NSE_EQ", "security_id": 11536},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ", "security_id": 1333},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ", "security_id": 1594},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ", "security_id": 1330},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ", "security_id": 1394},
    "ITC": {"symbol": "ITC", "segment": "NSE_EQ", "security_id": 1660},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ", "security_id": 3045},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ", "security_id": 392},
    "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ", "security_id": 1922},
    "LT": {"symbol": "LT", "segment": "NSE_EQ", "security_id": 2672},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ", "security_id": 5900},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ", "security_id": 317},
    "ASIANPAINT": {"symbol": "ASIANPAINT", "segment": "NSE_EQ", "security_id": 236},
    "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ", "security_id": 10999},
    "HCLTECH": {"symbol": "HCLTECH", "segment": "NSE_EQ", "security_id": 7229},
    "WIPRO": {"symbol": "WIPRO", "segment": "NSE_EQ", "security_id": 3787},
    "TITAN": {"symbol": "TITAN", "segment": "NSE_EQ", "security_id": 3506},
    "ULTRACEMCO": {"symbol": "ULTRACEMCO", "segment": "NSE_EQ", "security_id": 11532},
    "NESTLEIND": {"symbol": "NESTLEIND", "segment": "NSE_EQ", "security_id": 17963},
    "SUNPHARMA": {"symbol": "SUNPHARMA", "segment": "NSE_EQ", "security_id": 3351},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ", "security_id": 3456},
    "TATASTEEL": {"symbol": "TATASTEEL", "segment": "NSE_EQ", "security_id": 3499},
    "BAJAJFINSV": {"symbol": "BAJAJFINSV", "segment": "NSE_EQ", "security_id": 4598},
    "ONGC": {"symbol": "ONGC", "segment": "NSE_EQ", "security_id": 2475},
    "NTPC": {"symbol": "NTPC", "segment": "NSE_EQ", "security_id": 11630},
    "POWERGRID": {"symbol": "POWERGRID", "segment": "NSE_EQ", "security_id": 2752},
    "M&M": {"symbol": "M&M", "segment": "NSE_EQ", "security_id": 2031},
    "JSWSTEEL": {"symbol": "JSWSTEEL", "segment": "NSE_EQ", "security_id": 6733},
    "INDUSINDBK": {"symbol": "INDUSINDBK", "segment": "NSE_EQ", "security_id": 5258},
    "TECHM": {"symbol": "TECHM", "segment": "NSE_EQ", "security_id": 13538},
    "HINDALCO": {"symbol": "HINDALCO", "segment": "NSE_EQ", "security_id": 1363},
    "COALINDIA": {"symbol": "COALINDIA", "segment": "NSE_EQ", "security_id": 5215},
    "ADANIENT": {"symbol": "ADANIENT", "segment": "NSE_EQ", "security_id": 25},
    "ADANIPORTS": {"symbol": "ADANIPORTS", "segment": "NSE_EQ", "security_id": 31},
    "TRENT": {"symbol": "TRENT", "segment": "NSE_EQ", "security_id": 1964},
    "DLF": {"symbol": "DLF", "segment": "NSE_EQ", "security_id": 966},
    "GRASIM": {"symbol": "GRASIM", "segment": "NSE_EQ", "security_id": 1232},
    "DIVISLAB": {"symbol": "DIVISLAB", "segment": "NSE_EQ", "security_id": 10940},
    "CIPLA": {"symbol": "CIPLA", "segment": "NSE_EQ", "security_id": 694},
    "DRREDDY": {"symbol": "DRREDDY", "segment": "NSE_EQ", "security_id": 3721},
    "EICHERMOT": {"symbol": "EICHERMOT", "segment": "NSE_EQ", "security_id": 1023},
    "HEROMOTOCO": {"symbol": "HEROMOTOCO", "segment": "NSE_EQ", "security_id": 1348},
    "BRITANNIA": {"symbol": "BRITANNIA", "segment": "NSE_EQ", "security_id": 547},
    "TATACONSUM": {"symbol": "TATACONSUM", "segment": "NSE_EQ", "security_id": 3432},
    "GODREJCP": {"symbol": "GODREJCP", "segment": "NSE_EQ", "security_id": 7713},
    "DABUR": {"symbol": "DABUR", "segment": "NSE_EQ", "security_id": 881},
    "BEL": {"symbol": "BEL", "segment": "NSE_EQ", "security_id": 383},
    "SIEMENS": {"symbol": "SIEMENS", "segment": "NSE_EQ", "security_id": 3150},
    "BPCL": {"symbol": "BPCL", "segment": "NSE_EQ", "security_id": 500},
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
        
        # Session for connection pooling (FASTER API calls!)
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info("Bot initialized successfully")
    
    # ═══════════════════════════════════════════════════════════
    # TIMEZONE HELPERS
    # ═══════════════════════════════════════════════════════════
    
    def get_ist_time(self):
        """Current Indian time घेतो"""
        return datetime.now(IST)
    
    def format_ist_time(self, dt=None):
        """IST time format करतो"""
        if dt is None:
            dt = self.get_ist_time()
        return dt.strftime('%Y-%m-%d %H:%M:%S IST')
    
    # ═══════════════════════════════════════════════════════════
    # DATA FETCHING METHODS
    # ═══════════════════════════════════════════════════════════
    
    def get_historical_candles(self, security_id, segment, symbol):
        """Last 50 x 5-minute candles घेतो"""
        try:
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
            
            # Use session for faster calls
            response = self.session.post(
                DHAN_INTRADAY_URL,
                json=payload,
                timeout=10  # Reduced from 15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Different possible timestamp fields check करतो
                timestamp_field = None
                for field in ['start_Time', 'timestamp', 'time', 'start_time']:
                    if field in data:
                        timestamp_field = field
                        break
                
                if not timestamp_field:
                    logger.debug(f"{symbol}: No timestamp field, using index")
                    timestamps = [f"{i:04d}" for i in range(len(data.get('open', [])))]
                else:
                    timestamps = data[timestamp_field]
                
                # Arrays check करतो
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
                
                result = candles[-50:] if len(candles) > 50 else candles
                return result
            
            return None
            
        except Exception as e:
            logger.debug(f"{symbol}: Candles error: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Option chain data घेतो"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment,
                "Expiry": expiry
            }
            
            # Use session for faster calls
            response = self.session.post(
                DHAN_OPTION_CHAIN_URL,
                json=payload,
                timeout=10  # Reduced from 15
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if response has data
                if result.get('status') == 'success' and result.get('data'):
                    return result.get('data')
                else:
                    logger.debug(f"  Option chain error: {result.get('remarks', 'No data')}")
                    return None
            else:
                logger.debug(f"  Option chain HTTP error: {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.debug(f"  Option chain exception: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """सर्वात जवळचा expiry काढतो (AUTO SELECTION)"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment
            }
            
            # Use session for faster calls
            response = self.session.post(
                DHAN_EXPIRY_LIST_URL,
                json=payload,
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug: Response format check करतो
                logger.debug(f"  Expiry API Response: {data}")
                
                # Multiple formats handle करतो
                expiry_list = None
                
                # Format 1: {'status': 'success', 'data': ['2025-10-17', '2025-10-24']}
                if isinstance(data, dict) and data.get('status') == 'success':
                    if isinstance(data.get('data'), list) and len(data['data']) > 0:
                        expiry_list = data['data']
                
                # Format 2: {'data': ['2025-10-17', '2025-10-24']}
                elif isinstance(data, dict) and isinstance(data.get('data'), list):
                    expiry_list = data['data']
                
                # Format 3: ['2025-10-17', '2025-10-24']
                elif isinstance(data, list):
                    expiry_list = data
                
                if expiry_list and len(expiry_list) > 0:
                    nearest_expiry = expiry_list[0]
                    logger.debug(f"  Available expiries: {expiry_list[:3]}")  # First 3
                    logger.debug(f"  Selected (nearest): {nearest_expiry}")
                    return nearest_expiry
                else:
                    logger.warning(f"  No expiry list found in response")
                    return None
            else:
                logger.debug(f"  Expiry API HTTP error: {response.status_code}")
                return None
            
        except Exception as e:
            logger.debug(f"  Expiry API exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    # ═══════════════════════════════════════════════════════════
    # ATM STRIKE FILTERING
    # ═══════════════════════════════════════════════════════════
    
    def filter_atm_strikes(self, oc_data, spot_price):
        """ATM च्या आसपास फक्त ±5 strikes filter करतो (11 strikes total)"""
        try:
            all_strikes = oc_data.get('oc', {})
            if not all_strikes:
                logger.warning("  ⚠️ No strikes found in option chain")
                return {}
            
            # सगळे strikes numeric format मध्ये घेतो (handle both int and float keys)
            strike_map = {}  # float_strike -> original_key mapping
            strike_prices = []
            
            for strike_key in all_strikes.keys():
                try:
                    # Try converting strike key to float
                    strike_float = float(strike_key)
                    strike_prices.append(strike_float)
                    strike_map[strike_float] = strike_key
                except:
                    logger.debug(f"  Skipping invalid strike: {strike_key}")
                    continue
            
            if not strike_prices:
                logger.warning("  ⚠️ No valid strike prices found")
                return {}
            
            strike_prices.sort()
            
            # ATM strike शोधतो (spot च्या सर्वात जवळचा)
            atm_strike = min(strike_prices, key=lambda x: abs(x - spot_price))
            atm_index = strike_prices.index(atm_strike)
            
            # ATM ±5 strikes घेतो (total 11)
            start_idx = max(0, atm_index - 5)
            end_idx = min(len(strike_prices), atm_index + 6)
            
            filtered_strikes = strike_prices[start_idx:end_idx]
            
            # Filtered option chain तयार करतो (using original keys)
            filtered_oc = {
                'last_price': oc_data.get('last_price'),
                'oc': {}
            }
            
            for strike_float in filtered_strikes:
                original_key = strike_map.get(strike_float)
                if original_key and original_key in all_strikes:
                    filtered_oc['oc'][original_key] = all_strikes[original_key]
            
            logger.info(f"  ├─ ATM Strike: ₹{atm_strike:.0f}")
            logger.info(f"  ├─ Filtered: {len(filtered_oc['oc'])} strikes (ATM ±5)")
            if filtered_strikes:
                logger.info(f"  └─ Range: ₹{min(filtered_strikes):.0f} to ₹{max(filtered_strikes):.0f}")
            
            return filtered_oc
            
        except Exception as e:
            logger.error(f"  ❌ Error filtering ATM strikes: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return oc_data  # Return original data if filtering fails
    
    # ═══════════════════════════════════════════════════════════
    # BOT'S OWN ANALYSIS ENGINE
    # ═══════════════════════════════════════════════════════════
    
    def calculate_oi_changes(self, current_oc, symbol):
        """OI changes calculate करतो (previous vs current)"""
        try:
            if not self.redis_client:
                return {}
            
            # Previous OI Redis मधून घेतो
            prev_key = f"oi:current:{symbol}"
            prev_data = self.redis_client.get(prev_key)
            
            if not prev_data:
                # पहिल्यांदा run होतोय, current data save करतो
                self.redis_client.setex(
                    prev_key,
                    3600,  # 1 hour TTL
                    json.dumps(current_oc)
                )
                return {}
            
            prev_oc = json.loads(prev_data)
            
            # Changes calculate करतो
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
            
            # Current data save करतो
            self.redis_client.setex(prev_key, 3600, json.dumps(current_oc))
            
            return oi_changes
            
        except Exception as e:
            logger.error(f"Error calculating OI changes: {e}")
            return {}
    
    def detect_candlestick_pattern(self, candles):
        """Candlestick pattern detect करतो"""
        if len(candles) < 3:
            return None
        
        c1, c2, c3 = candles[-3:]
        
        # Bullish Engulfing
        if (c1['close'] < c1['open'] and  # Prev bearish
            c2['close'] > c2['open'] and  # Current bullish
            c2['open'] < c1['close'] and
            c2['close'] > c1['open']):
            return 'bullish_engulfing'
        
        # Bearish Engulfing
        if (c1['close'] > c1['open'] and
            c2['close'] < c2['open'] and
            c2['open'] > c1['close'] and
            c2['close'] < c1['open']):
            return 'bearish_engulfing'
        
        # Hammer (Bullish)
        body = abs(c2['close'] - c2['open'])
        lower_shadow = min(c2['open'], c2['close']) - c2['low']
        upper_shadow = c2['high'] - max(c2['open'], c2['close'])
        
        if lower_shadow > body * 2 and upper_shadow < body * 0.3:
            return 'hammer'
        
        # Shooting Star (Bearish)
        if upper_shadow > body * 2 and lower_shadow < body * 0.3:
            return 'shooting_star'
        
        return None
    
    def calculate_ema(self, candles, period):
        """EMA calculate करतो"""
        closes = [c['close'] for c in candles]
        
        if len(closes) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = closes[0]
        
        for price in closes[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_support_resistance(self, candles):
        """Support/Resistance calculate करतो"""
        if len(candles) < 20:
            return None, None
        
        highs = [c['high'] for c in candles[-20:]]
        lows = [c['low'] for c in candles[-20:]]
        
        resistance = max(highs)
        support = min(lows)
        
        return support, resistance
    
    def calculate_signal_score(self, oi_changes, candles, spot_price, oc_data):
        """Bot's own scoring algorithm"""
        try:
            score = 0
            signal_type = None
            details = {}
            
            # ═══════════════════════════════════
            # OI ANALYSIS (50 points max)
            # ═══════════════════════════════════
            oi_score = 0
            
            # PCR calculation
            total_ce_oi = sum(data.get('ce', {}).get('oi', 0) 
                            for data in oc_data.get('oc', {}).values())
            total_pe_oi = sum(data.get('pe', {}).get('oi', 0) 
                            for data in oc_data.get('oc', {}).values())
            
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
            details['pcr'] = round(pcr, 2)
            
            # PCR-based signal (15 points)
            if pcr < 0.8:
                oi_score += 15
                signal_type = "BEARISH"
            elif pcr > 1.2:
                oi_score += 15
                signal_type = "BULLISH"
            elif pcr < 0.9 or pcr > 1.1:
                oi_score += 8
            
            # Significant OI changes (20 points)
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
            
            # ═══════════════════════════════════
            # CHART ANALYSIS (50 points max)
            # ═══════════════════════════════════
            chart_score = 0
            
            # Pattern detection (20 points)
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
            
            # Volume confirmation (10 points)
            if len(candles) >= 10:
                last_volume = candles[-1]['volume']
                avg_volume = sum(c['volume'] for c in candles[-10:-1]) / 9
                
                if last_volume > avg_volume * 1.5:
                    chart_score += 10
                elif last_volume > avg_volume * 1.2:
                    chart_score += 5
            
            # Support/Resistance (10 points)
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
            
            # Moving averages (10 points)
            ema9 = self.calculate_ema(candles, 9)
            ema21 = self.calculate_ema(candles, 21)
            
            if ema9 and ema21:
                if ema9 > ema21 and signal_type == "BULLISH":
                    chart_score += 10
                elif ema9 < ema21 and signal_type == "BEARISH":
                    chart_score += 10
            
            chart_score = min(chart_score, 50)
            
            # ═══════════════════════════════════
            # FINAL SCORE
            # ═══════════════════════════════════
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
            logger.error(f"Error in signal scoring: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════
    # AI VERIFICATION
    # ═══════════════════════════════════════════════════════════
    
    def prepare_ai_data(self, symbol, spot, oi_changes, candles, bot_analysis):
        """AI साठी data prepare करतो"""
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
                'time': candle['timestamp'][-8:],
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
        """DeepSeek AI verification (FASTER!)"""
        try:
            prompt = f"""Expert Indian options trader. Verify signal quickly.

BOT: {ai_data['bot_analysis']['signal']} ({ai_data['bot_analysis']['score']}/100)
Symbol: {ai_data['symbol']} | Spot: ₹{ai_data['spot_price']}
PCR: {ai_data['oi_data']['pcr']} | Pattern: {ai_data['pattern']}

Respond JSON ONLY:
{{"status":"CONFIRMED/REJECTED","entry":24870,"target":24950,"stop_loss":24820,"risk_reward":1.6,"confidence":85,"reasoning":["Point 1","Point 2"]}}"""

            headers_ai = {
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.3,
                'max_tokens': 300  # Limit for faster response
            }
            
            response = requests.post(
                DEEPSEEK_API_URL,
                json=payload,
                headers=headers_ai,
                timeout=15  # Reduced from 30
            )
            
            if response.status_code == 200:
                ai_response = response.json()
                content = ai_response['choices'][0]['message']['content']
                
                # Parse JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
                
                return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.debug(f"AI error: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════
    # TELEGRAM MESSAGING
    # ═══════════════════════════════════════════════════════════
    
    async def send_signal(self, symbol, ai_data, ai_response, bot_analysis):
        """Final signal Telegram वर पाठवतो"""
        try:
            if ai_response['status'] == 'REJECTED':
                msg = f"❌ REJECTED - {symbol}\nBot: {bot_analysis['total_score']}/100\nReason: {ai_response['reasoning'][0]}"
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg
                )
                return
            
            signal_emoji = "🟢" if bot_analysis['signal_type'] == 'BULLISH' else "🔴"
            
            msg = f"""{signal_emoji} CONFIRMED TRADE SIGNAL {signal_emoji}

━━━━━━━━━━━━━━━━━━━━
📊 {symbol}
━━━━━━━━━━━━━━━━━━━━

⏰ {ai_data['timestamp']}
💰 Spot: ₹{ai_data['spot_price']:,.2f}

Direction: {bot_analysis['signal_type']} {'⬆️' if bot_analysis['signal_type']=='BULLISH' else '⬇️'}

━━━━━━━━━━━━━━━━━━━━
🎯 TRADE SETUP
━━━━━━━━━━━━━━━━━━━━

📍 Entry: ₹{ai_response['entry']:,.0f}
🎯 Target: ₹{ai_response['target']:,.0f} (+{ai_response['target']-ai_response['entry']:.0f})
🛑 SL: ₹{ai_response['stop_loss']:,.0f} (-{ai_response['entry']-ai_response['stop_loss']:.0f})
⚡ R:R: 1:{ai_response['risk_reward']:.1f}

━━━━━━━━━━━━━━━━━━━━
🤖 SCORES
━━━━━━━━━━━━━━━━━━━━

Bot Score: {bot_analysis['total_score']}/100
  ├─ OI: {bot_analysis['oi_score']}/50
  └─ Chart: {bot_analysis['chart_score']}/50

AI Confidence: {ai_response['confidence']}%
Rating: {'⭐' * (ai_response['confidence']//20)}

━━━━━━━━━━━━━━━━━━━━
📈 OI ANALYSIS
━━━━━━━━━━━━━━━━━━━━

PCR: {ai_data['oi_data']['pcr']}

Key Changes:
"""
            
            for change in ai_data['oi_data']['significant_changes'][:3]:
                msg += f"  {change['strike']}: {change['type']} {change['change_k']:+}K\n"
            
            msg += f"""
━━━━━━━━━━━━━━━━━━━━
💡 REASONING
━━━━━━━━━━━━━━━━━━━━

"""
            for i, reason in enumerate(ai_response['reasoning'][:3], 1):
                msg += f"{i}. {reason}\n"
            
            msg += f"""
━━━━━━━━━━━━━━━━━━━━
⚠️ Risk: {self.get_risk_level(bot_analysis['total_score'], ai_response['confidence'])}
🕒 Valid: Next 30-45 min
━━━━━━━━━━━━━━━━━━━━
"""
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg
            )
            
            # Save to Redis
            self.save_signal_to_redis(symbol, ai_data, ai_response, bot_analysis)
            
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
    
    def get_risk_level(self, bot_score, ai_conf):
        """Risk level calculate करतो"""
        avg = (bot_score + ai_conf) / 2
        if avg >= 80:
            return "LOW ✅"
        elif avg >= 65:
            return "MEDIUM ⚠️"
        else:
            return "HIGH ⛔"
    
    def save_signal_to_redis(self, symbol, ai_data, ai_response, bot_analysis):
        """Signal Redis मध्ये save करतो"""
        if not self.redis_client:
            return
        
        try:
            timestamp = self.get_ist_time().strftime("%Y%m%d_%H%M%S")
            signal_key = f"signal:{symbol}:{timestamp}"
            
            signal_data = {
                'symbol': symbol,
                'timestamp': ai_data['timestamp'],
                'bot_score': bot_analysis['total_score'],
                'ai_confidence': ai_response['confidence'],
                'signal': bot_analysis['signal_type'],
                'entry': ai_response['entry'],
                'target': ai_response['target'],
                'sl': ai_response['stop_loss'],
                'status': 'ACTIVE'
            }
            
            self.redis_client.setex(
                signal_key,
                86400,  # 24 hours
                json.dumps(signal_data)
            )
            
        except Exception as e:
            logger.error(f"Error saving to Redis: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # MAIN ANALYSIS FLOW
    # ═══════════════════════════════════════════════════════════
    
    async def analyze_symbol(self, symbol, info):
        """Single symbol चं complete analysis (OPTIMIZED!)"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"🔍 Analyzing: {symbol} | {self.format_ist_time()}")
            logger.info(f"{'─'*60}")
            
            # Step 1: Expiry घेतो (AUTO NEAREST)
            expiry = self.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"  ❌ {symbol}: No expiry available")
                return
            
            logger.info(f"  ✅ Nearest Expiry (AUTO): {expiry}")
            
            # Step 2: Option chain घेतो
            oc_data = self.get_option_chain(security_id, segment, expiry)
            if not oc_data:
                logger.warning(f"  ❌ {symbol}: No option chain data")
                return
            
            spot_price = oc_data.get('last_price', 0)
            logger.info(f"  ✅ Spot Price: ₹{spot_price:,.2f}")
            logger.info(f"  ✅ Total Strikes: {len(oc_data.get('oc', {}))}")
            
            # Step 2.5: ATM ±5 strikes filter करतो (NEW!)
            filtered_oc = self.filter_atm_strikes(oc_data, spot_price)
            
            # Step 3: Candles घेतो
            candles = self.get_historical_candles(security_id, segment, symbol)
            if not candles or len(candles) < 20:
                logger.warning(f"  ❌ {symbol}: Not enough candles (Got: {len(candles) if candles else 0}, Need: 20)")
                return
            
            logger.info(f"  ✅ Candles: {len(candles)} × 5min")
            
            # Last 3 candles info
            if len(candles) >= 3:
                last_candle = candles[-1]
                prev_candle = candles[-2]
                logger.info(f"  ├─ Current: O:{last_candle['open']:.1f} H:{last_candle['high']:.1f} L:{last_candle['low']:.1f} C:{last_candle['close']:.1f} V:{last_candle['volume']:,}")
                logger.info(f"  └─ Previous: O:{prev_candle['open']:.1f} H:{prev_candle['high']:.1f} L:{prev_candle['low']:.1f} C:{prev_candle['close']:.1f} V:{prev_candle['volume']:,}")
            
            # Step 4: OI changes calculate करतो
            oi_changes = self.calculate_oi_changes(filtered_oc, symbol)
            if not oi_changes:
                logger.info(f"  ℹ️ {symbol}: First run - saving baseline data")
                return
            
            # OI changes summary
            total_ce_change = sum(c.get('ce_oi_change', 0) for c in oi_changes.values())
            total_pe_change = sum(c.get('pe_oi_change', 0) for c in oi_changes.values())
            logger.info(f"  ✅ OI Changes:")
            logger.info(f"    ├─ CE Total: {total_ce_change:+,.0f}")
            logger.info(f"    └─ PE Total: {total_pe_change:+,.0f}")
            
            # Step 5: Bot's Analysis
            logger.info(f"  🤖 Running bot analysis...")
            bot_analysis = self.calculate_signal_score(
                oi_changes, candles, spot_price, filtered_oc
            )
            
            if not bot_analysis:
                logger.warning(f"  ❌ {symbol}: Bot analysis failed")
                return
            
            logger.info(f"  📊 Bot Score: {bot_analysis['total_score']}/100")
            logger.info(f"    ├─ OI Score: {bot_analysis['oi_score']}/50")
            logger.info(f"    ├─ Chart Score: {bot_analysis['chart_score']}/50")
            logger.info(f"    ├─ Signal: {bot_analysis['signal_type']}")
            logger.info(f"    ├─ Pattern: {bot_analysis['details'].get('pattern', 'None')}")
            logger.info(f"    ├─ PCR: {bot_analysis['details'].get('pcr', 0):.2f}")
            logger.info(f"    └─ Confidence: {bot_analysis['confidence']}")
            
            # Step 6: Check if score >= 70
            if not bot_analysis['send_to_ai']:
                logger.info(f"  ⚠️ {symbol}: Score too low, skipping AI")
                return
            
            # 🔥 HIGH SCORE DETECTED!
            logger.info(f"\n{'='*60}")
            logger.info(f"🔥🔥🔥 HIGH SCORE ALERT! 🔥🔥🔥")
            logger.info(f"{'='*60}")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Score: {bot_analysis['total_score']}/100")
            logger.info(f"Signal: {bot_analysis['signal_type']}")
            logger.info(f"Spot: ₹{spot_price:,.2f}")
            logger.info(f"{'='*60}")
            
            # Step 7: Prepare data for AI
            ai_data = self.prepare_ai_data(
                symbol, spot_price, oi_changes, candles, bot_analysis
            )
            
            # Step 8: AI Verification
            logger.info(f"🤖 Sending to DeepSeek AI for verification...")
            ai_response = await self.verify_with_ai(ai_data)
            
            if not ai_response:
                logger.warning(f"❌ AI verification failed or unavailable")
                return
            
            logger.info(f"✅ AI Response: {ai_response['status']}")
            logger.info(f"✅ AI Confidence: {ai_response.get('confidence', 0)}%")
            
            # Step 9: Send signal to Telegram
            await self.send_signal(symbol, ai_data, ai_response, bot_analysis)
            
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Error - {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # ═══════════════════════════════════════════════════════════
    # MAIN RUN LOOP
    # ═══════════════════════════════════════════════════════════
    
    async def analyze_batch_parallel(self, batch):
        """Batch मधले सगळे symbols parallel process करतो (FAST!)"""
        tasks = []
        for symbol in batch:
            info = STOCKS_INDICES[symbol]
            tasks.append(self.analyze_symbol(symbol, info))
        
        # सगळे parallel run करतो
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def run(self):
        """Main bot loop - every 5 minutes"""
        logger.info("🚀 Smart Trading Bot Started!")
        
        # Startup message
        await self.send_startup_message()
        
        # Symbol batches (10 at a time for parallel processing)
        all_symbols = list(STOCKS_INDICES.keys())
        batch_size = 10  # Increased from 5 to 10
        batches = [all_symbols[i:i+batch_size] 
                  for i in range(0, len(all_symbols), batch_size)]
        
        logger.info(f"📊 Tracking {len(all_symbols)} symbols in {len(batches)} batches")
        logger.info(f"⚡ PARALLEL MODE: 10 stocks per batch!")
        logger.info(f"⏱️ Estimated cycle time: {len(batches) * 30} seconds")
        
        while self.running:
            try:
                cycle_start = self.get_ist_time()
                logger.info(f"\n{'#'*80}")
                logger.info(f"CYCLE START: {self.format_ist_time(cycle_start)}")
                logger.info(f"{'#'*80}\n")
                
                # Process each batch (PARALLEL!)
                for batch_num, batch in enumerate(batches, 1):
                    batch_start = self.get_ist_time()
                    logger.info(f"\n📦 Batch {batch_num}/{len(batches)}: {batch}")
                    
                    # PARALLEL PROCESSING! 🚀
                    await self.analyze_batch_parallel(batch)
                    
                    batch_duration = (self.get_ist_time() - batch_start).total_seconds()
                    logger.info(f"✅ Batch {batch_num} completed in {batch_duration:.1f}s")
                    
                    # Minimal wait between batches (Dhan rate limit protection)
                    if batch_num < len(batches):
                        logger.info(f"⏸️ Waiting 3 seconds before next batch...")
                        await asyncio.sleep(3)  # Reduced from 5 to 3
                
                cycle_end = self.get_ist_time()
                duration = (cycle_end - cycle_start).total_seconds()
                
                # Dynamic sleep calculation
                target_cycle = 300  # 5 minutes
                sleep_time = max(30, target_cycle - duration)  # Min 30 sec sleep
                
                logger.info(f"\n{'#'*80}")
                logger.info(f"CYCLE COMPLETE!")
                logger.info(f"├─ Duration: {duration:.1f}s / {target_cycle}s")
                logger.info(f"├─ Sleeping: {sleep_time:.0f}s")
                next_cycle_time = self.get_ist_time() + timedelta(seconds=sleep_time)
                logger.info(f"└─ Next cycle: {next_cycle_time.strftime('%I:%M:%S %p IST')}")
                logger.info(f"{'#'*80}\n")
                
                await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup message"""
        try:
            msg = """🤖 SMART TRADING BOT STARTED!

━━━━━━━━━━━━━━━━━━━━
📊 FEATURES
━━━━━━━━━━━━━━━━━━━━

✅ Bot's Own Analysis Engine
  ├─ OI Analysis (PCR, Changes)
  ├─ Chart Patterns (50×5min candles)
  ├─ Volume Analysis
  ├─ Support/Resistance
  └─ ATM ±5 Strikes (11 total)

✅ AI Verification (DeepSeek)
  ├─ Only High-Score Signals
  └─ Entry/Target/SL

✅ Smart Filtering
  ├─ Score >= 70/100
  ├─ Auto Nearest Expiry
  └─ AI Confirms

━━━━━━━━━━━━━━━━━━━━
📈 TRACKING
━━━━━━━━━━━━━━━━━━━━

Symbols: """ + str(len(STOCKS_INDICES)) + """
Update: Every 5 minutes
Batches: """ + str((len(STOCKS_INDICES) + 9) // 10) + """
Timezone: IST (Indian Time)

━━━━━━━━━━━━━━━━━━━━
⚡ POWERED BY
━━━━━━━━━━━━━━━━━━━━

🔹 DhanHQ API v2
🔹 DeepSeek AI
🔹 Redis Cache
🔹 Railway.app

━━━━━━━━━━━━━━━━━━━━
Let's catch some trades! 🎯
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
        required_vars = [
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID,
            DHAN_CLIENT_ID,
            DHAN_ACCESS_TOKEN,
            DEEPSEEK_API_KEY
        ]
        
        if not all(required_vars):
            logger.error("❌ Missing environment variables!")
            logger.error("Required: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, DEEPSEEK_API_KEY")
            exit(1)
        
        logger.info("✅ All environment variables found!")
        
        bot = SmartTradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
