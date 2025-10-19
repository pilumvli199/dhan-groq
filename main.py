"""
🤖 SMART MONEY F&O TRADING BOT - FIXED VERSION
Complete integration: Dhan API + DeepSeek AI + Auto Expiry Selection

Author: Trading Bot Team
Version: 2.1 - FIXED STARTUP BUG
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
    
    # Stocks/Indices to track
    SYMBOLS = {
        # Indices
        "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
        "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
        
        # Top Stocks
        "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
        "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
        "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
        "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
        "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
        "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
        "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ"},
        "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ"},
    }


# ========================
# DHAN API HANDLER
# ========================
class DhanAPI:
    """Dhan HQ API Integration"""
    
    def __init__(self):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.security_id_map = {}
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
            csv_data = response.text.split('\n')
            
            for symbol, info in Config.SYMBOLS.items():
                segment = info['segment']
                symbol_name = info['symbol']
                
                reader = csv.DictReader(csv_data)
                
                for row in reader:
                    try:
                        # Index साठी
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
                                    logger.info(f"✅ {symbol}: Security ID = {sec_id}")
                                    break
                        
                        # Stock साठी
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
                                    break
                    except Exception as e:
                        continue
                
                # Reset reader
                csv_data = response.text.split('\n')
            
            logger.info(f"🎯 Total {len(self.security_id_map)} securities loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading security IDs: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        """Get nearest expiry for options"""
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
                if data.get('status') == 'success' and data.get('data'):
                    expiries = data['data']
                    if expiries:
                        nearest = expiries[0]
                        logger.info(f"✅ Nearest expiry: {nearest}")
                        return nearest
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting expiry: {e}")
            return None
    
    def get_historical_candles(self, security_id: int, segment: str, symbol: str) -> Optional[pd.DataFrame]:
        """Get last 5 days of 5-minute candles"""
        try:
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
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
                Config.DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'open' in data and 'close' in data:
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data.get('start_Time', [])),
                        'open': data.get('open', []),
                        'high': data.get('high', []),
                        'low': data.get('low', []),
                        'close': data.get('close', []),
                        'volume': data.get('volume', [])
                    })
                    
                    logger.info(f"✅ {symbol}: Fetched {len(df)} candles")
                    return df
            
            logger.warning(f"⚠️ {symbol}: No candle data")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching candles for {symbol}: {e}")
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str) -> Optional[Dict]:
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
                    logger.info(f"✅ Option chain data received")
                    return data['data']
            
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
        
        # 1. HAMMER
        if current['range'] > 0:
            if (current['lower_wick'] > current['body'] * 2 and 
                current['upper_wick'] < current['body'] * 0.5):
                patterns.append({
                    'name': 'HAMMER',
                    'type': 'BULLISH',
                    'confidence': 70
                })
        
        # 2. SHOOTING STAR
        if current['range'] > 0:
            if (current['upper_wick'] > current['body'] * 2 and 
                current['lower_wick'] < current['body'] * 0.5):
                patterns.append({
                    'name': 'SHOOTING_STAR',
                    'type': 'BEARISH',
                    'confidence': 70
                })
        
        # 3. BULLISH ENGULFING
        if (not prev['is_bullish'] and current['is_bullish'] and
            c3['close'] > c2['open'] and c3['open'] < c2['close']):
            patterns.append({
                'name': 'BULLISH_ENGULFING',
                'type': 'BULLISH',
                'confidence': 80
            })
        
        # 4. BEARISH ENGULFING
        if (prev['is_bullish'] and not current['is_bullish'] and
            c3['close'] < c2['open'] and c3['open'] > c2['close']):
            patterns.append({
                'name': 'BEARISH_ENGULFING',
                'type': 'BEARISH',
                'confidence': 80
            })
        
        # 5. DOJI
        if current['range'] > 0 and current['body'] < current['range'] * 0.1:
            patterns.append({
                'name': 'DOJI',
                'type': 'NEUTRAL',
                'confidence': 60
            })
        
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
        
        return support, resistance
    
    @staticmethod
    def analyze_oi(option_chain: Dict) -> Dict:
        """Analyze Open Interest"""
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
            
            if pcr > 1.5:
                signal = "BULLISH"
                confidence = 75
            elif pcr < 0.5:
                signal = "BEARISH"
                confidence = 75
            else:
                signal = "NEUTRAL"
                confidence = 50
            
            return {
                'pcr': pcr,
                'signal': signal,
                'confidence': confidence,
                'call_oi': total_call_oi,
                'put_oi': total_put_oi
            }
        
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
        return ratio


# ========================
# DEEPSEEK AI ANALYZER
# ========================
class DeepSeekAnalyzer:
    """DeepSeek V3 AI Integration"""
    
    @staticmethod
    def analyze(context: Dict) -> Optional[Dict]:
        """Get AI analysis"""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
तुम expert F&O trader हो। Data analyze करो आणि signal दो।

DATA:
- Symbol: {context['symbol']}
- Spot Price: ₹{context['spot_price']}
- Support: ₹{context.get('support', 'N/A')}
- Resistance: ₹{context.get('resistance', 'N/A')}
- Patterns: {context['patterns']}
- PCR: {context['pcr']}
- OI Signal: {context['oi_signal']}
- Volume Ratio: {context['volume_ratio']}x
- Confluence Score: {context['confluence_score']}/10

JSON मध्ये answer:
{{
  "signal": "BUY/SELL/WAIT",
  "confidence": 75,
  "entry": 42150,
  "target": 42300,
  "stop_loss": 42050,
  "risk_reward": "1:2",
  "marathi_explanation": "BANKNIFTY ला strong support मिळाला आहे..."
}}
"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an expert F&O trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                logger.info(f"✅ AI: {analysis['signal']} ({analysis['confidence']}%)")
                return analysis
            
            return None
        
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            return None


# ========================
# CHART GENERATOR
# ========================
class ChartGenerator:
    """Generate PNG charts"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, symbol: str, spot_price: float, 
                     support: float, resistance: float, patterns: List) -> Optional[io.BytesIO]:
        """Create candlestick chart with mplfinance"""
        try:
            if len(df) < 2:
                return None
            
            chart_df = df.copy()
            chart_df.set_index('timestamp', inplace=True)
            chart_df = chart_df[['open', 'high', 'low', 'close', 'volume']]
            chart_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            mc = mpf.make_marketcolors(
                up='#26a69a',
                down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#333333',
                facecolor='#1e1e1e',
                figcolor='#1e1e1e',
                gridaxis='both',
                y_on_right=False
            )
            
            hlines = []
            if support:
                hlines.append(support)
            if resistance:
                hlines.append(resistance)
            
            fig, axes = mpf.plot(
                chart_df.tail(100),
                type='candle',
                style=s,
                volume=True,
                hlines=dict(hlines=hlines, colors=['blue', 'red'], linestyle='--'),
                title=f'\n{symbol} | Spot: ₹{spot_price:,.2f}',
                ylabel='Price (₹)',
                ylabel_lower='Volume',
                figsize=(14, 8),
                returnfig=True,
                tight_layout=True
            )
            
            axes[0].set_title(
                f'{symbol} | Spot: ₹{spot_price:,.2f} | Patterns: {len(patterns)}',
                color='white',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            for ax in axes:
                ax.tick_params(colors='white', which='both')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            buf.seek(0)
            plt.close(fig)
            
            logger.info("✅ Chart generated")
            return buf
            
        except Exception as e:
            logger.error(f"❌ Chart error: {e}")
            return None


# ========================
# MAIN BOT
# ========================
class SmartMoneyBot:
    """Main Trading Bot"""
    
    def __init__(self):
        logger.info("🔧 Initializing SmartMoneyBot...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.dhan = DhanAPI()
        self.running = True
        logger.info("✅ SmartMoneyBot initialized")
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        if now.weekday() >= 5:
            logger.info(f"📅 Weekend: Market closed")
            return False
        
        if Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE:
            return True
        
        logger.info(f"⏰ Outside market hours: {current_time}")
        return False
    
    async def scan_symbol(self, symbol: str, info: Dict):
        """Scan single symbol"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*50}")
            logger.info(f"🔍 Scanning: {symbol}")
            logger.info(f"{'='*50}")
            
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No expiry found")
                return
            
            logger.info(f"📅 Expiry: {expiry}")
            
            candles_df = self.dhan.get_historical_candles(security_id, segment, symbol)
            if candles_df is None or len(candles_df) < 20:
                logger.warning(f"⚠️ {symbol}: Insufficient candle data")
                return
            
            option_chain = self.dhan.get_option_chain(security_id, segment, expiry)
            if not option_chain:
                logger.warning(f"⚠️ {symbol}: No option chain data")
                return
            
            spot_price = option_chain.get('last_price', 0)
            
            patterns = PatternDetector.detect_patterns(candles_df)
            support, resistance = SmartMoneyAnalyzer.calculate_support_resistance(candles_df)
            oi_analysis = SmartMoneyAnalyzer.analyze_oi(option_chain)
            volume_ratio = SmartMoneyAnalyzer.calculate_volume_ratio(candles_df)
            
            confluence = len(patterns)
            if support and resistance:
                confluence += 1
            if volume_ratio > 1.5:
                confluence += 1
            if oi_analysis['confidence'] > 70:
                confluence += 1
            confluence = min(confluence + 4, 10)
            
            context = {
                'symbol': symbol,
                'spot_price': spot_price,
                'support': support,
                'resistance': resistance,
                'patterns': [p['name'] for p in patterns],
                'pcr': oi_analysis['pcr'],
                'oi_signal': oi_analysis['signal'],
                'volume_ratio': round(volume_ratio, 2),
                'confluence_score': confluence
            }
            
            analysis = DeepSeekAnalyzer.analyze(context)
            
            if not analysis:
                logger.warning(f"⚠️ {symbol}: No AI analysis")
                return
            
            if analysis['confidence'] < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"⏸️ {symbol}: Confidence too low ({analysis['confidence']}%)")
                return
            
            chart_buf = ChartGenerator.create_chart(
                candles_df, symbol, spot_price, support, resistance, patterns
            )
            
            signal_emoji = "🟢" if analysis['signal'] == "BUY" else "🔴" if analysis['signal'] == "SELL" else "⚪"
            
            message = f"""
🚀 <b>SMART MONEY SIGNAL</b>

📊 Symbol: <b>{symbol}</b>
💰 Spot: ₹{spot_price:,.2f}
📅 Expiry: {expiry}
⏰ Time: {datetime.now().strftime('%H:%M:%S IST')}

{signal_emoji} Signal: <b>{analysis['signal']}</b>
💪 Confidence: <b>{analysis['confidence']}%</b>

🎯 <b>TRADE SETUP:</b>
• Entry: ₹{analysis.get('entry', spot_price)}
• Target: ₹{analysis.get('target', spot_price * 1.02)}
• Stop-Loss: ₹{analysis.get('stop_loss', spot_price * 0.98)}
• Risk/Reward: {analysis.get('risk_reward', '1:2')}

📈 <b>ANALYSIS:</b>
• Support: ₹{support:,.0f if support else 'N/A'}
• Resistance: ₹{resistance:,.0f if resistance else 'N/A'}
• PCR: {oi_analysis['pcr']:.2f}
• Volume: {volume_ratio:.2f}x avg
• Patterns: {len(patterns)}
• Confluence: {confluence}/10

📝 <b>Marathi Explanation:</b>
{analysis.get('marathi_explanation', 'Analysis pending...')}

⚡ Disclaimer: Trade at your own risk.
"""
            
            if chart_buf:
                await self.bot.send_photo(
                    chat_id=Config.TELEGRAM_CHAT_ID,
                    photo=chart_buf,
                    caption=f"📊 {symbol} - Candlestick Chart"
                )
                logger.info(f"✅ {symbol}: Chart sent")
                await asyncio.sleep(1)
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info(f"✅ {symbol}: Alert sent!")
            
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_startup_message(self):
        """Send startup notification"""
        try:
            logger.info("📤 Sending startup message to Telegram...")
            msg = f"""
🤖 <b>Smart Money Bot Started!</b>

📊 Tracking: {len(self.dhan.security_id_map)} symbols
⏰ Scan Interval: {Config.SCAN_INTERVAL//60} minutes
🎯 Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}%
⏱️ Market Hours: {Config.MARKET_OPEN} - {Config.MARKET_CLOSE}

🔍 <b>Features:</b>
✓ Auto nearest expiry selection
✓ Historical 5-min candles analysis
✓ Smart Money Concepts (OI, PCR, Volume)
✓ 10 Candlestick patterns detection
✓ DeepSeek V3 AI reasoning
✓ Text + PNG chart alerts

📈 <b>Symbols:</b>
{', '.join(self.dhan.security_id_map.keys())}

⚡ Powered by: DhanHQ API + DeepSeek AI
🚂 Status: <b>ACTIVE</b> ✅
⏰ Startup: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("✅ Startup message sent to Telegram successfully!")
        except Exception as e:
            logger.error(f"❌ Startup message error: {e}")
            logger.error(traceback.format_exc())
    
    async def run(self):
        """Main bot loop"""
        logger.info("="*60)
        logger.info("🚀 SMART MONEY BOT STARTING...")
        logger.info("="*60)
        
        # Validate credentials
        logger.info("🔐 Validating API credentials...")
        if not all([Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID, 
                    Config.DHAN_CLIENT_ID, Config.DHAN_ACCESS_TOKEN, Config.DEEPSEEK_API_KEY]):
            logger.error("❌ Missing API credentials! Check environment variables:")
            logger.error(f"   - TELEGRAM_BOT_TOKEN: {'✅' if Config.TELEGRAM_BOT_TOKEN else '❌'}")
            logger.error(f"   - TELEGRAM_CHAT_ID: {'✅' if Config.TELEGRAM_CHAT_ID else '❌'}")
            logger.error(f"   - DHAN_CLIENT_ID: {'✅' if Config.DHAN_CLIENT_ID else '❌'}")
            logger.error(f"   - DHAN_ACCESS_TOKEN: {'✅' if Config.DHAN_ACCESS_TOKEN else '❌'}")
            logger.error(f"   - DEEPSEEK_API_KEY: {'✅' if Config.DEEPSEEK_API_KEY else '❌'}")
            return
        
        logger.info("✅ All credentials validated!")
        
        # Load security IDs
        logger.info("📥 Loading security IDs from Dhan...")
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("❌ Failed to load securities. Exiting...")
            return
        
        logger.info(f"✅ Loaded {len(self.dhan.security_id_map)} securities successfully!")
        
        # Send startup message
        await self.send_startup_message()
        
        logger.info("="*60)
        logger.info("🎯 Bot is now RUNNING! Waiting for market hours...")
        logger.info("="*60)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("😴 Market closed. Sleeping for 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Starting scan cycle at {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Scan each symbol
                for symbol, info in self.dhan.security_id_map.items():
                    await self.scan_symbol(symbol, info)
                    await asyncio.sleep(3)  # Rate limit
                
                logger.info(f"\n✅ Scan complete! Waiting {Config.SCAN_INTERVAL//60} minutes...")
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
        logger.info("🚀 INITIALIZING SMART MONEY BOT v2.1")
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
    logger.info("="*60)
    logger.info("🎬 STARTING BOT...")
    logger.info(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info("="*60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        logger.error(traceback.format_exc())
