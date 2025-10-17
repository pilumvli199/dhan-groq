import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime
import logging
import csv
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplfinance as mpf
import pandas as pd
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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# DeepSeek API URL
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"

# Stock/Index List
STOCKS_INDICES = {
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
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
        logger.info("Bot initialized successfully")
    
    async def load_security_ids(self):
        """Dhan मधून security IDs load करतो"""
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
                                        logger.info(f"✅ {symbol}: Security ID = {sec_id}")
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
                                        break
                        except Exception as e:
                            continue
                    
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
    
    def get_historical_data(self, security_id, segment, symbol, candle_count=50):
        """Last 50 5-minute candles घेतो"""
        try:
            from datetime import datetime, timedelta
            
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
            
            logger.info(f"Intraday API call for {symbol}: {payload}")
            
            response = requests.post(
                DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'open' in data and 'high' in data and 'low' in data and 'close' in data:
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
                    
                    last_candles = candles[-candle_count:] if len(candles) >= candle_count else candles
                    logger.info(f"{symbol}: Returning last {len(last_candles)} candles (5 min)")
                    return last_candles
                else:
                    logger.warning(f"{symbol}: Invalid response format")
                    return None
            
            logger.warning(f"{symbol}: Historical data नाही मिळाला - Status: {response.status_code}")
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
                logger.warning(f"{symbol}: Not enough candles ({len(df)}) for chart")
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
    
    def prepare_analysis_data(self, symbol, candles, oc_data, expiry):
        """DeepSeek साठी data prepare करतो"""
        try:
            spot_price = oc_data.get('last_price', 0)
            oc = oc_data.get('oc', {})
            
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            atm_idx = strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(strikes), atm_idx + 6)
            selected_strikes = strikes[start_idx:end_idx]
            
            option_data = []
            for strike in selected_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                option_data.append({
                    'strike': strike,
                    'is_atm': strike == atm_strike,
                    'ce': {
                        'ltp': ce.get('last_price', 0),
                        'oi': ce.get('oi', 0),
                        'volume': ce.get('volume', 0),
                        'iv': ce.get('implied_volatility', 0),
                        'delta': ce.get('greeks', {}).get('delta', 0),
                        'theta': ce.get('greeks', {}).get('theta', 0)
                    },
                    'pe': {
                        'ltp': pe.get('last_price', 0),
                        'oi': pe.get('oi', 0),
                        'volume': pe.get('volume', 0),
                        'iv': pe.get('implied_volatility', 0),
                        'delta': pe.get('greeks', {}).get('delta', 0),
                        'theta': pe.get('greeks', {}).get('theta', 0)
                    }
                })
            
            candle_data = []
            for c in candles[-50:]:
                candle_data.append({
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'volume': c['volume']
                })
            
            analysis_input = {
                'symbol': symbol,
                'spot_price': spot_price,
                'expiry': expiry,
                'atm_strike': atm_strike,
                'candles': candle_data,
                'option_chain': option_data
            }
            
            return analysis_input
            
        except Exception as e:
            logger.error(f"Error preparing analysis data: {e}")
            return None
    
    async def get_deepseek_analysis(self, analysis_data):
        """DeepSeek V3 कडून AI analysis घेतो"""
        try:
            if not DEEPSEEK_API_KEY:
                logger.error("DeepSeek API key missing!")
                return None
            
            prompt = f"""You are an expert options trader analyzing Indian stock market data.

Symbol: {analysis_data['symbol']}
Spot Price: ₹{analysis_data['spot_price']:,.2f}
ATM Strike: ₹{analysis_data['atm_strike']:,.0f}
Expiry: {analysis_data['expiry']}

CANDLESTICK DATA (Last 50 5-min candles):
{json.dumps(analysis_data['candles'], indent=2)}

OPTION CHAIN DATA (ATM ± 5 strikes):
{json.dumps(analysis_data['option_chain'], indent=2)}

Task: Analyze the data and provide ONLY ONE clear trading signal:
1. BUY CE (Call Option) - if bullish setup
2. BUY PE (Put Option) - if bearish setup
3. NO TRADE - if no clear signal

Your response MUST be in this EXACT JSON format:
{{
  "signal": "BUY CE" or "BUY PE" or "NO TRADE",
  "strike": <recommended strike price>,
  "entry_price": <option premium to enter>,
  "target": <profit target premium>,
  "stoploss": <stoploss premium>,
  "confidence": <0-100>,
  "reasoning": "<brief 2-3 line explanation>"
}}

Consider:
- Candlestick patterns (bullish/bearish)
- Support/Resistance levels
- Volume trends
- OI data (buildup/unwinding)
- IV levels
- Greeks (Delta, Theta)

Be strict: Only give BUY signal if confidence > 70%."""

            headers = {
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,
                'max_tokens': 500
            }
            
            logger.info(f"Calling DeepSeek API for {analysis_data['symbol']}...")
            
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = content[start:end]
                        analysis = json.loads(json_str)
                        logger.info(f"✅ DeepSeek analysis received for {analysis_data['symbol']}")
                        return analysis
                    else:
                        logger.warning("No JSON found in DeepSeek response")
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    logger.error(f"Response content: {content}")
                    return None
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return None
    
    def format_signal_message(self, symbol, analysis):
        """Trading signal साठी message format"""
        try:
            signal = analysis.get('signal', 'NO TRADE')
            
            if signal == 'NO TRADE':
                msg = f"⚪️ *{symbol} - NO TRADE SIGNAL*\n\n"
                msg += f"📊 Confidence: {analysis.get('confidence', 0)}%\n"
                msg += f"💭 Reason: {analysis.get('reasoning', 'Market conditions unclear')}\n"
                return msg
            
            emoji = "🟢" if signal == "BUY CE" else "🔴"
            
            msg = f"{emoji} *{symbol} - {signal}*\n\n"
            msg += f"🎯 Strike: ₹{analysis.get('strike', 0):,.0f}\n"
            msg += f"💰 Entry: ₹{analysis.get('entry_price', 0):,.2f}\n"
            msg += f"📈 Target: ₹{analysis.get('target', 0):,.2f}\n"
            msg += f"🛑 Stoploss: ₹{analysis.get('stoploss', 0):,.2f}\n"
            msg += f"📊 Confidence: {analysis.get('confidence', 0)}%\n\n"
            msg += f"💡 *Analysis:*\n{analysis.get('reasoning', 'AI analysis completed')}\n\n"
            msg += f"⚠️ _Disclaimer: This is AI-generated analysis. Trade at your own risk._"
            
            return msg
            
        except Exception as e:
            logger.error(f"Error formatting signal message: {e}")
            return None
    
    async def send_option_chain_batch(self, symbols_batch):
        """एका batch चे option chain + AI analysis पाठवतो"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    logger.warning(f"Skipping {symbol} - No security ID")
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                
                expiry = self.get_nearest_expiry(security_id, segment)
                if not expiry:
                    logger.warning(f"{symbol}: Expiry नाही मिळाला")
                    continue
                
                logger.info(f"Fetching data for {symbol} (Expiry: {expiry})...")
                
                oc_data = self.get_option_chain(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"{symbol}: Option chain data नाही मिळाला")
                    continue
                
                spot_price = oc_data.get('last_price', 0)
                
                logger.info(f"Fetching last 50 candles for {symbol}...")
                candles = self.get_historical_data(security_id, segment, symbol, candle_count=50)
                
                if not candles or len(candles) < 10:
                    logger.warning(f"{symbol}: Insufficient candle data")
                    continue
                
                chart_buf = self.create_candlestick_chart(candles, symbol, spot_price)
                
                logger.info(f"🤖 Starting AI analysis for {symbol}...")
                analysis_input = self.prepare_analysis_data(symbol, candles, oc_data, expiry)
                
                if analysis_input:
                    ai_analysis = await self.get_deepseek_analysis(analysis_input)
                    
                    if ai_analysis:
                        if chart_buf:
                            await self.bot.send_photo(
                                chat_id=TELEGRAM_CHAT_ID,
                                photo=chart_buf,
                                caption=f"📊 {symbol} - Last {len(candles)} Candles"
                            )
                            await asyncio.sleep(1)
                        
                        signal_msg = self.format_signal_message(symbol, ai_analysis)
                        if signal_msg:
                            await self.bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                text=signal_msg,
                                parse_mode='Markdown'
                            )
                            logger.info(f"✅ {symbol} AI signal sent: {ai_analysis.get('signal')}")
                    else:
                        logger.warning(f"{symbol}: AI analysis failed")
                else:
                    logger.warning(f"{symbol}: Analysis data preparation failed")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                await asyncio.sleep(3)
    
    async def run(self):
        """Main loop - every 5 minutes option analysis + signals"""
        logger.info("🚀 Bot started with DeepSeek AI! Loading security IDs...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("Failed to load security IDs. Exiting...")
            return
        
        await self.send_startup_message()
        
        all_symbols = list(self.security_id_map.keys())
        batch_size = 5
        batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
        
        logger.info(f"Total {len(all_symbols)} symbols in {len(batches)} batches")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*50}")
                logger.info(f"Starting AI analysis cycle at {timestamp}")
                logger.info(f"{'='*50}")
                
                for batch_num, batch in enumerate(batches, 1):
                    logger.info(f"\n📦 Processing Batch {batch_num}/{len(batches)}: {batch}")
                    await self.send_option_chain_batch(batch)
                    
                    if batch_num < len(batches):
                        logger.info(f"Waiting 5 seconds before next batch...")
                        await asyncio.sleep(5)
                
                logger.info("\n✅ All batches completed!")
                logger.info("⏳ Waiting 5 minutes for next cycle...\n")
                
                await asyncio.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Bot सुरू झाल्यावर message"""
        try:
            msg = "🤖 *Dhan Option Chain Bot with DeepSeek AI Started!*\n\n"
            msg += f"📊 Tracking {len(self.security_id_map)} stocks/indices\n"
            msg += "⏱️ Updates every 5 minutes\n"
            msg += "🤖 AI-Powered Features:\n"
            msg += "  • DeepSeek V3 Analysis\n"
            msg += "  • CE/PE Buy Signals\n"
            msg += "  • Entry/Target/Stoploss\n"
            msg += "  • Candlestick Charts (50 candles)\n"
            msg += "  • Option Chain + Greeks\n\n"
            msg += "✅ Powered by DhanHQ + DeepSeek AI\n"
            msg += "🚂 Deployed on Railway.app\n\n"
            msg += "_Market Hours: 9:15 AM - 3:30 PM (Mon-Fri)_"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("Startup message sent")
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")


if __name__ == "__main__":
    try:
        required_vars = {
            'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
            'DHAN_CLIENT_ID': DHAN_CLIENT_ID,
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN,
            'DEEPSEEK_API_KEY': DEEPSEEK_API_KEY
        }
        
        missing_vars = [k for k, v in required_vars.items() if not v]
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            logger.error("Please set all required variables:")
            logger.error("  - TELEGRAM_BOT_TOKEN")
            logger.error("  - TELEGRAM_CHAT_ID")
            logger.error("  - DHAN_CLIENT_ID")
            logger.error("  - DHAN_ACCESS_TOKEN")
            logger.error("  - DEEPSEEK_API_KEY")
            exit(1)
        
        logger.info("🚀 Starting Dhan Option Chain Bot with DeepSeek AI...")
        bot = DhanOptionChainBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
