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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ALL STOCKS LIST (Extended)
STOCKS_INDICES = {
    # Indices
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I", "security_id": 13},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I", "security_id": 25},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I", "security_id": 51},
    "FINNIFTY": {"symbol": "FINNIFTY", "segment": "IDX_I", "security_id": 27},
    
    # Large Cap Stocks
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ", "security_id": 2885},
    "TCS": {"symbol": "TCS", "segment": "NSE_EQ", "security_id": 11536},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ", "security_id": 1333},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ", "security_id": 1330},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ", "security_id": 80},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ", "security_id": 1594},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ", "security_id": 1394},
    "ITC": {"symbol": "ITC", "segment": "NSE_EQ", "security_id": 1660},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ", "security_id": 3045},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ", "security_id": 317},
    "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ", "security_id": 1922},
    "LT": {"symbol": "LT", "segment": "NSE_EQ", "security_id": 2600},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ", "security_id": 5900},
    "ASIANPAINT": {"symbol": "ASIANPAINT", "segment": "NSE_EQ", "security_id": 235},
    "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ", "security_id": 10999},
    "HCLTECH": {"symbol": "HCLTECH", "segment": "NSE_EQ", "security_id": 7229},
    "WIPRO": {"symbol": "WIPRO", "segment": "NSE_EQ", "security_id": 3787},
    "ULTRACEMCO": {"symbol": "ULTRACEMCO", "segment": "NSE_EQ", "security_id": 11532},
    "TITAN": {"symbol": "TITAN", "segment": "NSE_EQ", "security_id": 3506},
    "SUNPHARMA": {"symbol": "SUNPHARMA", "segment": "NSE_EQ", "security_id": 3351},
    "NTPC": {"symbol": "NTPC", "segment": "NSE_EQ", "security_id": 11630},
    "NESTLEIND": {"symbol": "NESTLEIND", "segment": "NSE_EQ", "security_id": 17963},
    "POWERGRID": {"symbol": "POWERGRID", "segment": "NSE_EQ", "security_id": 2752},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ", "security_id": 3456},
    "TATASTEEL": {"symbol": "TATASTEEL", "segment": "NSE_EQ", "security_id": 3499},
    "M&M": {"symbol": "M&M", "segment": "NSE_EQ", "security_id": 2031},
    "JSWSTEEL": {"symbol": "JSWSTEEL", "segment": "NSE_EQ", "security_id": 6733},
    "HINDALCO": {"symbol": "HINDALCO", "segment": "NSE_EQ", "security_id": 1363},
    "COALINDIA": {"symbol": "COALINDIA", "segment": "NSE_EQ", "security_id": 5215},
    "ONGC": {"symbol": "ONGC", "segment": "NSE_EQ", "security_id": 2475},
    "ADANIPORTS": {"symbol": "ADANIPORTS", "segment": "NSE_EQ", "security_id": 15083},
    "TECHM": {"symbol": "TECHM", "segment": "NSE_EQ", "security_id": 13538},
    "DRREDDY": {"symbol": "DRREDDY", "segment": "NSE_EQ", "security_id": 3721},
    "BRITANNIA": {"symbol": "BRITANNIA", "segment": "NSE_EQ", "security_id": 547},
    "EICHERMOT": {"symbol": "EICHERMOT", "segment": "NSE_EQ", "security_id": 910},
    "DIVISLAB": {"symbol": "DIVISLAB", "segment": "NSE_EQ", "security_id": 10940},
    
    # Mid Cap Stocks
    "ADANIENT": {"symbol": "ADANIENT", "segment": "NSE_EQ", "security_id": 25},
    "TRENT": {"symbol": "TRENT", "segment": "NSE_EQ", "security_id": 1964},
    "DMART": {"symbol": "DMART", "segment": "NSE_EQ", "security_id": 14977},
    "PERSISTENT": {"symbol": "PERSISTENT", "segment": "NSE_EQ", "security_id": 18365},
    "LTIM": {"symbol": "LTIM", "segment": "NSE_EQ", "security_id": 17818},
    "LICI": {"symbol": "LICI", "segment": "NSE_EQ", "security_id": 21808},
    "ZOMATO": {"symbol": "ZOMATO", "segment": "NSE_EQ", "security_id": 14418},
    "PAYTM": {"symbol": "PAYTM", "segment": "NSE_EQ", "security_id": 13404},
    "NYKAA": {"symbol": "NYKAA", "segment": "NSE_EQ", "security_id": 20287},
    "BAJAJFINSV": {"symbol": "BAJAJFINSV", "segment": "NSE_EQ", "security_id": 4963},
    "SHRIRAMFIN": {"symbol": "SHRIRAMFIN", "segment": "NSE_EQ", "security_id": 3153},
    "SBILIFE": {"symbol": "SBILIFE", "segment": "NSE_EQ", "security_id": 21808},
    "HDFCLIFE": {"symbol": "HDFCLIFE", "segment": "NSE_EQ", "security_id": 467},
    "VEDL": {"symbol": "VEDL", "segment": "NSE_EQ", "security_id": 3063},
    "GODREJCP": {"symbol": "GODREJCP", "segment": "NSE_EQ", "security_id": 10099},
    "INDIGO": {"symbol": "INDIGO", "segment": "NSE_EQ", "security_id": 11195},
    "GRASIM": {"symbol": "GRASIM", "segment": "NSE_EQ", "security_id": 1232},
    "ADANIGREEN": {"symbol": "ADANIGREEN", "segment": "NSE_EQ", "security_id": 21808},
    "BAJAJ-AUTO": {"symbol": "BAJAJ-AUTO", "segment": "NSE_EQ", "security_id": 16669},
    "SIEMENS": {"symbol": "SIEMENS", "segment": "NSE_EQ", "security_id": 3150},
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
        
        logger.info("Bot initialized successfully")
    
    # ═══════════════════════════════════════════════════════════
    # DATA FETCHING METHODS
    # ═══════════════════════════════════════════════════════════
    
    def get_historical_candles(self, security_id, segment, symbol):
        """Last 50 x 5-minute candles घेतो"""
        try:
            exch_seg = "IDX_I" if segment == "IDX_I" else "NSE_EQ"
            instrument = "INDEX" if segment == "IDX_I" else "EQUITY"
            
            to_date = datetime.now()
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
                DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                candles = []
                for i in range(len(data.get('open', []))):
                    candles.append({
                        'timestamp': data['start_Time'][i],
                        'open': data['open'][i],
                        'high': data['high'][i],
                        'low': data['low'][i],
                        'close': data['close'][i],
                        'volume': data['volume'][i]
                    })
                
                # Last 50 candles return करतो
                return candles[-50:] if len(candles) > 50 else candles
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting candles for {symbol}: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Option chain data घेतो"""
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
                return response.json().get('data')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """सर्वात जवळचा expiry काढतो"""
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
            logger.error(f"Error getting expiry: {e}")
            return None
    
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
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
            prompt = f"""You are an expert Indian options trader. Verify this signal.

BOT DETECTED: {ai_data['bot_analysis']['signal']} Signal
Score: {ai_data['bot_analysis']['score']}/100

Symbol: {ai_data['symbol']}
Spot: ₹{ai_data['spot_price']}

OI Data:
PCR: {ai_data['oi_data']['pcr']}
Changes: {json.dumps(ai_data['oi_data']['significant_changes'])}

Pattern: {ai_data['pattern']}
Support: {ai_data['support']}
Resistance: {ai_data['resistance']}

Last 5 Candles:
{json.dumps(ai_data['candles'][-5:])}

Task: Verify signal. Respond in JSON:
{{
  "status": "CONFIRMED" or "REJECTED",
  "entry": 24870,
  "target": 24950,
  "stop_loss": 24820,
  "risk_reward": 1.6,
  "confidence": 85,
  "reasoning": ["Point 1", "Point 2"]
}}"""

            headers_ai = {
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.3
            }
            
            response = requests.post(
                DEEPSEEK_API_URL,
                json=payload,
                headers=headers_ai,
                timeout=30
            )
            
            if response.status_code == 200:
                ai_response = response.json()
                content = ai_response['choices'][0]['message']['content']
                
                # Parse JSON from response
                return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.error(f"AI verification error: {e}")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        """Single symbol चं complete analysis"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing: {symbol}")
            logger.info(f"{'='*60}")
            
            # Step 1: Expiry घेतो
            expiry = self.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"{symbol}: No expiry found")
                return
            
            # Step 2: Option chain घेतो
            oc_data = self.get_option_chain(security_id, segment, expiry)
            if not oc_data:
                logger.warning(f"{symbol}: No option chain data")
                return
            
            spot_price = oc_data.get('last_price', 0)
            
            # Step 3: Candles घेतो
            candles = self.get_historical_candles(security_id, segment, symbol)
            if not candles or len(candles) < 20:
                logger.warning(f"{symbol}: Not enough candles")
                return
            
            # Step 4: OI changes calculate करतो
            oi_changes = self.calculate_oi_changes(oc_data, symbol)
            if not oi_changes:
                logger.info(f"{symbol}: First run, no OI changes yet")
                return
            
            # Step 5: Bot's Analysis
            logger.info(f"{symbol}: Running bot analysis...")
            bot_analysis = self.calculate_signal_score(
                oi_changes, candles, spot_price, oc_data
            )
            
            if not bot_analysis:
                return
            
            logger.info(f"{symbol}: Bot Score = {bot_analysis['total_score']}/100")
            logger.info(f"  ├─ OI Score: {bot_analysis['oi_score']}/50")
            logger.info(f"  ├─ Chart Score: {bot_analysis['chart_score']}/50")
            logger.info(f"  └─ Signal: {bot_analysis['signal_type']}")
            
            # Step 6: Check if score >= 70
            if not bot_analysis['send_to_ai']:
                logger.info(f"{symbol}: Score too low, skipping AI verification")
                return
            
            # Step 7: Prepare data for AI
            logger.info(f"{symbol}: Preparing data for AI...")
            ai_data = self.prepare_ai_data(
                symbol, spot_price, oi_changes, candles, bot_analysis
            )
            
            # Step 8: AI Verification
            logger.info(f"{symbol}: Sending to AI for verification...")
            ai_response = await self.verify_with_ai(ai_data)
            
            if not ai_response:
                logger.warning(f"{symbol}: AI verification failed")
                return
            
            logger.info(f"{symbol}: AI Status = {ai_response['status']}")
            logger.info(f"{symbol}: AI Confidence = {ai_response.get('confidence', 0)}%")
            
            # Step 9: Send signal to Telegram
            await self.send_signal(symbol, ai_data, ai_response, bot_analysis)
            
            logger.info(f"✅ {symbol}: Complete!")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # MAIN RUN LOOP
    # ═══════════════════════════════════════════════════════════
    
    async def run(self):
        """Main bot loop - every 5 minutes"""
        logger.info("🚀 Smart Trading Bot Started!")
        
        # Startup message
        await self.send_startup_message()
        
        # Symbol batches (5 at a time)
        all_symbols = list(STOCKS_INDICES.keys())
        batch_size = 5
        batches = [all_symbols[i:i+batch_size] 
                  for i in range(0, len(all_symbols), batch_size)]
        
        logger.info(f"📊 Tracking {len(all_symbols)} symbols in {len(batches)} batches")
        
        while self.running:
            try:
                cycle_start = datetime.now()
                logger.info(f"\n{'#'*80}")
                logger.info(f"CYCLE START: {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'#'*80}\n")
                
                # Process each batch
                for batch_num, batch in enumerate(batches, 1):
                    logger.info(f"\n📦 Batch {batch_num}/{len(batches)}: {batch}")
                    
                    for symbol in batch:
                        info = STOCKS_INDICES[symbol]
                        await self.analyze_symbol(symbol, info)
                        await asyncio.sleep(2)  # 2 sec between symbols
                    
                    # Wait between batches
                    if batch_num < len(batches):
                        logger.info(f"⏸️ Waiting 10 seconds before next batch...")
                        await asyncio.sleep(10)
                
                cycle_end = datetime.now()
                duration = (cycle_end - cycle_start).total_seconds()
                
                logger.info(f"\n{'#'*80}")
                logger.info(f"CYCLE COMPLETE: Duration = {duration:.1f}s")
                logger.info(f"Next cycle in 5 minutes...")
                logger.info(f"{'#'*80}\n")
                
                # Wait 5 minutes
                await asyncio.sleep(300)
                
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
  ├─ Chart Patterns
  ├─ Volume Analysis
  └─ Support/Resistance

✅ AI Verification (DeepSeek)
  ├─ Only High-Score Signals
  └─ Entry/Target/SL

✅ Smart Filtering
  ├─ Score >= 70/100
  └─ AI Confirms

━━━━━━━━━━━━━━━━━━━━
📈 TRACKING
━━━━━━━━━━━━━━━━━━━━

Symbols: """ + str(len(STOCKS_INDICES)) + """
Update: Every 5 minutes
Batches: """ + str((len(STOCKS_INDICES) + 4) // 5) + """

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
