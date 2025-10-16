import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime
import logging
import csv
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplfinance as mpf
import pandas as pd

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

# DeepSeek API
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"

# Stock/Index List - Symbol mapping
STOCKS_INDICES = {
    # Indices
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
    
    # Stocks
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
    "TECHM": {"symbol": "TECHM", "segment": "NSE_EQ"},
    "LICI": {"symbol": "LICI", "segment": "NSE_EQ"},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ"},
    "NTPC": {"symbol": "NTPC", "segment": "NSE_EQ"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ"},
    "POWERGRID": {"symbol": "POWERGRID", "segment": "NSE_EQ"},
    "ONGC": {"symbol": "ONGC", "segment": "NSE_EQ"},
    "PERSISTENT": {"symbol": "PERSISTENT", "segment": "NSE_EQ"},
    "DRREDDY": {"symbol": "DRREDDY", "segment": "NSE_EQ"},
    "M&M": {"symbol": "M&M", "segment": "NSE_EQ"},
    "WIPRO": {"symbol": "WIPRO", "segment": "NSE_EQ"},
    "DMART": {"symbol": "DMART", "segment": "NSE_EQ"},
    "TRENT": {"symbol": "TRENT", "segment": "NSE_EQ"},
}

# ========================
# DEEPSEEK AI HELPER
# ========================

class DeepSeekAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def compress_market_data(self, symbol, spot, candles, option_chain, expiry):
        """Data compress करतो - 70% tokens कमी!"""
        try:
            # Last 10 candles घेतो (full history नाही)
            recent_candles = candles[-10:] if len(candles) > 10 else candles
            
            # Candles compress करतो: [timestamp, o, h, l, c, v]
            c_data = []
            for candle in recent_candles:
                c_data.append([
                    candle.get('timestamp', '')[-8:],  # फक्त time (HH:MM:SS)
                    round(candle.get('open', 0), 1),
                    round(candle.get('high', 0), 1),
                    round(candle.get('low', 0), 1),
                    round(candle.get('close', 0), 1),
                    int(candle.get('volume', 0) / 1000)  # K मध्ये
                ])
            
            # Option chain compress करतो
            oc_data = option_chain.get('oc', {})
            strikes = sorted([float(s) for s in oc_data.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot))
            atm_idx = strikes.index(atm_strike)
            
            # ATM च्या आजूबाजूचे 5 strikes (एकूण 11)
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(strikes), atm_idx + 6)
            selected_strikes = strikes[start_idx:end_idx]
            
            # Option data compress: [strike, ce_ltp, ce_oi, ce_vol, pe_ltp, pe_oi, pe_vol]
            oc_compressed = []
            for strike in selected_strikes:
                sk = f"{strike:.6f}"
                sd = oc_data.get(sk, {})
                ce = sd.get('ce', {})
                pe = sd.get('pe', {})
                
                oc_compressed.append([
                    int(strike),
                    round(ce.get('last_price', 0), 1),
                    int(ce.get('oi', 0) / 1000),  # K मध्ये
                    int(ce.get('volume', 0) / 1000),
                    round(pe.get('last_price', 0), 1),
                    int(pe.get('oi', 0) / 1000),
                    int(pe.get('volume', 0) / 1000)
                ])
            
            # Compressed JSON
            compressed = {
                "s": symbol,  # symbol
                "p": round(spot, 2),  # spot price
                "e": expiry[-10:],  # expiry (short)
                "c": c_data,  # candles [time, o, h, l, c, v]
                "oc": oc_compressed,  # option chain
                "atm": int(atm_strike)
            }
            
            return compressed
            
        except Exception as e:
            logger.error(f"Compression error for {symbol}: {e}")
            return None
    
    def analyze_with_deepseek(self, compressed_data):
        """DeepSeek V3 AI analysis (compressed data साठी)"""
        try:
            symbol = compressed_data.get('s')
            spot = compressed_data.get('p')
            atm = compressed_data.get('atm')
            
            # Prompt तयार करतो (compact)
            prompt = f"""Analyze this market data (compressed format):
Symbol: {symbol}
Spot: ₹{spot}
ATM: ₹{atm}

Recent Candles (last 10): {json.dumps(compressed_data.get('c', []))}
Option Chain: {json.dumps(compressed_data.get('oc', []))}

Provide:
1. Trend (bullish/bearish/neutral) with reason
2. Key support/resistance levels
3. Option strategy suggestion (1-2 lines)
4. Risk level (low/medium/high)

Keep response under 150 words."""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert Indian stock market analyst. Provide concise, actionable insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 400,
                "stream": False
            }
            
            logger.info(f"Calling DeepSeek API for {symbol}...")
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis = data['choices'][0]['message']['content']
                tokens_used = data.get('usage', {})
                
                logger.info(f"✅ {symbol} analysis done - Tokens: {tokens_used}")
                return analysis
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek analysis error: {e}")
            return None

# ========================
# BOT CODE
# ========================

class DhanOptionChainBot:
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
        self.deepseek = DeepSeekAnalyzer(DEEPSEEK_API_KEY)
        logger.info("Bot initialized with DeepSeek AI")
    
    async def load_security_ids(self):
        """Dhan मधून security IDs load करतो (without pandas)"""
        try:
            logger.info("Loading security IDs from Dhan...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code == 200:
                # CSV parse करतो manually
                csv_data = response.text.split('\n')
                reader = csv.DictReader(csv_data)
                
                for symbol, info in STOCKS_INDICES.items():
                    segment = info['segment']
                    symbol_name = info['symbol']
                    
                    # CSV मध्ये शोधतो
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
                    
                    # Reset CSV reader for next symbol
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)
                
                logger.info(f"Total {len(self.security_id_map)} securities loaded")
                return True
            else:
                logger.error(f"Failed to load instruments: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading security IDs: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Last 5 days चे सर्व 5-minute candles घेतो"""
        try:
            from datetime import datetime, timedelta
            
            # Exchange segment निवडतो
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
            # Last 5 trading days साठी dates
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            # Intraday API साठी payload
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
                
                if 'open' in data and 'high' in data:
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
                            'open': opens[i] if i < len(opens) else 0,
                            'high': highs[i] if i < len(highs) else 0,
                            'low': lows[i] if i < len(lows) else 0,
                            'close': closes[i] if i < len(closes) else 0,
                            'volume': volumes[i] if i < len(volumes) else 0
                        })
                    
                    logger.info(f"{symbol}: {len(candles)} candles loaded")
                    return candles
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def create_candlestick_chart(self, candles, symbol, spot_price):
        """Candlestick chart तयार करतो"""
        try:
            df_data = []
            for candle in candles:
                timestamp = candle.get('timestamp', candle.get('start_Time', ''))
                df_data.append({
                    'Date': pd.to_datetime(timestamp) if timestamp else pd.Timestamp.now(),
                    'Open': float(candle.get('open', 0)),
                    'High': float(candle.get('high', 0)),
                    'Low': float(candle.get('low', 0)),
                    'Close': float(candle.get('close', 0)),
                    'Volume': int(float(candle.get('volume', 0)))
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            
            if len(df) < 2:
                return None
            
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
            
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f'\n{symbol} - Last {len(candles)} Candles | Spot: ₹{spot_price:,.2f}',
                ylabel='Price (₹)',
                ylabel_lower='Volume',
                figsize=(12, 8),
                returnfig=True,
                tight_layout=True
            )
            
            axes[0].set_title(
                f'{symbol} - Last {len(candles)} Candles | Spot: ₹{spot_price:,.2f}',
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
            
            return buf
            
        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
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
                    expiries = data['data']
                    if expiries:
                        return expiries[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting expiry: {e}")
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
                data = response.json()
                if data.get('data'):
                    return data['data']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return None
    
    def format_option_chain_message(self, symbol, data, expiry):
        """Option chain साठी सुंदर message format"""
        try:
            spot_price = data.get('last_price', 0)
            oc_data = data.get('oc', {})
            
            if not oc_data:
                return None
            
            strikes = sorted([float(s) for s in oc_data.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            atm_idx = strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(strikes), atm_idx + 6)
            selected_strikes = strikes[start_idx:end_idx]
            
            msg = f"📊 *{symbol} OPTION CHAIN*\n"
            msg += f"📅 Expiry: {expiry}\n"
            msg += f"💰 Spot: ₹{spot_price:,.2f}\n"
            msg += f"🎯 ATM: ₹{atm_strike:,.0f}\n\n"
            
            msg += "```\n"
            msg += "Strike   CE-LTP  CE-OI  CE-Vol  PE-LTP  PE-OI  PE-Vol\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            for strike in selected_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc_data.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                ce_ltp = ce.get('last_price', 0)
                ce_oi = ce.get('oi', 0)
                ce_vol = ce.get('volume', 0)
                
                pe_ltp = pe.get('last_price', 0)
                pe_oi = pe.get('oi', 0)
                pe_vol = pe.get('volume', 0)
                
                atm_mark = "🔸" if strike == atm_strike else "  "
                
                msg += f"{atm_mark}{strike:6.0f}  {ce_ltp:6.1f} {ce_oi/1000:6.0f}K {ce_vol/1000:6.0f}K  {pe_ltp:6.1f} {pe_oi/1000:6.0f}K {pe_vol/1000:6.0f}K\n"
            
            msg += "```\n"
            
            return msg
            
        except Exception as e:
            logger.error(f"Error formatting message for {symbol}: {e}")
            return None
    
    async def send_option_chain_batch(self, symbols_batch):
        """एका batch चे option chain + AI analysis पाठवतो"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                
                expiry = self.get_nearest_expiry(security_id, segment)
                if not expiry:
                    continue
                
                logger.info(f"Processing {symbol}...")
                
                # Data fetch करतो
                oc_data = self.get_option_chain(security_id, segment, expiry)
                if not oc_data:
                    continue
                
                spot_price = oc_data.get('last_price', 0)
                candles = self.get_historical_data(security_id, segment, symbol)
                
                # Chart पाठवतो
                chart_buf = None
                if candles:
                    chart_buf = self.create_candlestick_chart(candles, symbol, spot_price)
                
                if chart_buf:
                    await self.bot.send_photo(
                        chat_id=TELEGRAM_CHAT_ID,
                        photo=chart_buf,
                        caption=f"📊 {symbol} - Chart"
                    )
                    await asyncio.sleep(1)
                
                # Option chain message
                oc_message = self.format_option_chain_message(symbol, oc_data, expiry)
                if oc_message:
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=oc_message,
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(1)
                
                # 🤖 AI ANALYSIS (DeepSeek V3)
                if candles and oc_data:
                    logger.info(f"🤖 Getting AI analysis for {symbol}...")
                    
                    # Data compress करतो
                    compressed = self.deepseek.compress_market_data(
                        symbol, spot_price, candles, oc_data, expiry
                    )
                    
                    if compressed:
                        # AI analysis मिळवतो
                        analysis = self.deepseek.analyze_with_deepseek(compressed)
                        
                        if analysis:
                            ai_msg = f"🤖 *AI ANALYSIS - {symbol}*\n\n{analysis}\n\n"
                            ai_msg += "_Powered by DeepSeek V3_"
                            
                            await self.bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                text=ai_msg,
                                parse_mode='Markdown'
                            )
                            logger.info(f"✅ {symbol} AI analysis sent")
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                await asyncio.sleep(3)
    
    async def run(self):
        """Main loop"""
        logger.info("🚀 Bot starting with DeepSeek AI...")
        
        success = await self.load_security_ids()
        if not success:
            return
        
        await self.send_startup_message()
        
        all_symbols = list(self.security_id_map.keys())
        batch_size = 5
        batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*50}")
                logger.info(f"Update cycle: {timestamp}")
                logger.info(f"{'='*50}")
                
                for batch_num, batch in enumerate(batches, 1):
                    logger.info(f"\n📦 Batch {batch_num}/{len(batches)}: {batch}")
                    await self.send_option_chain_batch(batch)
                    
                    if batch_num < len(batches):
                        await asyncio.sleep(5)
                
                logger.info("\n✅ All batches completed!")
                logger.info("⏳ Next cycle in 5 minutes...\n")
                
                await asyncio.sleep(300)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup message"""
        try:
            msg = "🤖 *Dhan Option Chain Bot + AI Started!*\n\n"
            msg += f"📊 Tracking: {len(self.security_id_map)} stocks/indices\n"
            msg += "⏱️ Updates: Every 5 minutes\n"
            msg += "🤖 AI: DeepSeek V3 Analysis\n\n"
            msg += "📈 Features:\n"
            msg += "  • Candlestick Charts\n"
            msg += "  • Option Chain Data\n"
            msg += "  • AI Market Analysis\n"
            msg += "  • Trading Suggestions\n\n"
            msg += "✅ Ready to analyze markets!"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Startup message error: {e}")


if __name__ == "__main__":
    try:
        if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, 
                    DHAN_ACCESS_TOKEN, DEEPSEEK_API_KEY]):
            logger.error("❌ Missing environment variables!")
            logger.error("Required: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, " +
                        "DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, DEEPSEEK_API_KEY")
            exit(1)
        
        bot = DhanOptionChainBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
