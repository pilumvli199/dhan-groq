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

# AI API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# AI API URLs
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
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
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
    
    # Stocks
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    
    
}

# ========================
# AI HELPER CLASS
# ========================
class AIAnalyzer:
    """2-Layer AI Analysis System"""
    
    @staticmethod
    async def cerebras_filter(stocks_data):
        """Layer 1: Cerebras Llama 3.3 70B - Quick Filtering"""
        try:
            # Compact data format for API
            compact_data = []
            for stock in stocks_data:
                compact_data.append({
                    "sym": stock["symbol"],
                    "spot": stock["spot_price"],
                    "candles": stock["candles"][-60:],  # Last 60 only
                    "oc": {
                        "ce_oi": stock["option_chain"].get("ce_total_oi", 0),
                        "pe_oi": stock["option_chain"].get("pe_total_oi", 0),
                        "pcr": stock["option_chain"].get("pcr", 0),
                        "atm_ce": stock["option_chain"].get("atm_ce_ltp", 0),
                        "atm_pe": stock["option_chain"].get("atm_pe_ltp", 0)
                    }
                })
            
            prompt = f"""Analyze these {len(compact_data)} stocks and filter TOP 5 with best trading opportunities.

Data: {json.dumps(compact_data)}

Filter based on:
1. Price momentum (last 60 candles)
2. Volume surge
3. PCR ratio (Put-Call Ratio)
4. OI changes

Return ONLY JSON array with top 5 symbols: ["SYM1","SYM2","SYM3","SYM4","SYM5"]"""

            headers = {
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.3-70b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100
            }
            
            response = requests.post(CEREBRAS_API_URL, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract JSON array
                import re
                match = re.search(r'\[(.*?)\]', content)
                if match:
                    filtered = json.loads(f'[{match.group(1)}]')
                    logger.info(f"🤖 Cerebras filtered: {filtered}")
                    return filtered[:5]
            
            logger.warning("Cerebras filtering failed, returning first 5")
            return [s["symbol"] for s in stocks_data[:5]]
            
        except Exception as e:
            logger.error(f"Cerebras error: {e}")
            return [s["symbol"] for s in stocks_data[:5]]
    
    @staticmethod
    async def deepseek_v3_analysis(filtered_stocks_data):
        """Layer 2: DeepSeek V3 - Deep Analysis"""
        try:
            analyses = []
            
            for stock in filtered_stocks_data:
                # Compact format
                compact = {
                    "sym": stock["symbol"],
                    "spot": stock["spot_price"],
                    "candles": stock["candles"][-30:],  # Last 30 for analysis
                    "oc": stock["option_chain"]
                }
                
                prompt = f"""Deep technical analysis for {stock['symbol']}:

{json.dumps(compact, indent=2)}

Analyze:
1. Support/Resistance levels
2. Trend direction
3. Volume analysis
4. Option chain signals (PCR, OI buildup)
5. Risk level (1-10)

Return compact JSON:
{{"sym":"...","signal":"BUY/SELL/HOLD","conf":0-100,"risk":1-10,"reason":"..."}}"""

                headers = {
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "deepseek-chat",  # DeepSeek V3 model
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 200
                }
                
                response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Extract JSON
                    import re
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        analysis = json.loads(match.group(0))
                        analyses.append(analysis)
                        logger.info(f"🧠 DeepSeek V3: {analysis}")
                
                await asyncio.sleep(2)  # Rate limit
            
            return analyses
            
        except Exception as e:
            logger.error(f"DeepSeek V3 error: {e}")
            return []
    
    @staticmethod
    async def deepseek_r1_decision(deepseek_v3_analyses, full_data):
        """Layer 3: DeepSeek R1 - Final Decision (same API key, different model)"""
        try:
            prompt = f"""You are a professional trader. Make FINAL trading decisions.

DeepSeek V3 Analysis:
{json.dumps(deepseek_v3_analyses, indent=2)}

Full Market Context:
{json.dumps(full_data, indent=2)}

For each stock, give:
1. FINAL ACTION: BUY/SELL/HOLD
2. Entry Price
3. Target Price
4. Stop Loss
5. Confidence (0-100%)
6. Brief reasoning (max 50 words)

Return JSON array of decisions."""

            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",  # Same key
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-reasoner",  # DeepSeek R1 model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract JSON array
                import re
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    decisions = json.loads(match.group(0))
                    logger.info(f"🎯 DeepSeek R1 decisions: {len(decisions)}")
                    return decisions
            
            return []
            
        except Exception as e:
            logger.error(f"DeepSeek R1 error: {e}")
            return []

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
        self.ai_analyzer = AIAnalyzer()
        self.current_batch_index = 0  # Track current batch
        self.all_symbols = []  # Store all symbol names
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
    
    def get_historical_data(self, security_id, segment, symbol):
        """Last 60 candles घेतो (compact)"""
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
                            'open': opens[i],
                            'high': highs[i],
                            'low': lows[i],
                            'close': closes[i],
                            'volume': volumes[i]
                        })
                    
                    # Last 60 candles only
                    return candles[-60:] if len(candles) > 60 else candles
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """सर्वात जवळचा expiry"""
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
        """Option chain data (compact)"""
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
    
    def compact_option_chain(self, oc_data):
        """Option chain को compact format मध्ये convert"""
        try:
            spot_price = oc_data.get('last_price', 0)
            oc = oc_data.get('oc', {})
            
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            ce = atm_data.get('ce', {})
            pe = atm_data.get('pe', {})
            
            # Calculate totals
            ce_total_oi = sum(oc.get(f"{s:.6f}", {}).get('ce', {}).get('oi', 0) for s in strikes)
            pe_total_oi = sum(oc.get(f"{s:.6f}", {}).get('pe', {}).get('oi', 0) for s in strikes)
            
            return {
                "spot": spot_price,
                "atm": atm_strike,
                "atm_ce_ltp": ce.get('last_price', 0),
                "atm_pe_ltp": pe.get('last_price', 0),
                "ce_total_oi": ce_total_oi,
                "pe_total_oi": pe_total_oi,
                "pcr": pe_total_oi / ce_total_oi if ce_total_oi > 0 else 0,
                "ce_iv": ce.get('implied_volatility', 0),
                "pe_iv": pe.get('implied_volatility', 0)
            }
            
        except Exception as e:
            logger.error(f"Error compacting option chain: {e}")
            return {}
    
    async def collect_all_data(self):
        """सर्व stocks चा data collect करतो"""
        all_data = []
        
        for symbol in self.security_id_map.keys():
            try:
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                
                expiry = self.get_nearest_expiry(security_id, segment)
                if not expiry:
                    continue
                
                candles = self.get_historical_data(security_id, segment, symbol)
                oc_data = self.get_option_chain(security_id, segment, expiry)
                
                if candles and oc_data:
                    compact_oc = self.compact_option_chain(oc_data)
                    
                    all_data.append({
                        "symbol": symbol,
                        "spot_price": compact_oc.get("spot", 0),
                        "candles": candles,
                        "option_chain": compact_oc,
                        "expiry": expiry
                    })
                    
                    logger.info(f"✅ Collected: {symbol}")
                
                await asyncio.sleep(3)  # Dhan rate limit
                
            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}")
        
        return all_data
    
    async def ai_scan_and_alert(self):
        """AI-powered scanning आणि alerts"""
        try:
            logger.info("🤖 Starting AI scanning...")
            
            # Step 1: सर्व data collect करतो
            all_data = await self.collect_all_data()
            logger.info(f"📊 Collected {len(all_data)} stocks data")
            
            if len(all_data) < 5:
                logger.warning("Not enough data for AI analysis")
                return
            
            # Step 2: Cerebras filtering (Top 5)
            filtered_symbols = await self.ai_analyzer.cerebras_filter(all_data)
            filtered_data = [d for d in all_data if d["symbol"] in filtered_symbols]
            logger.info(f"🔍 Cerebras filtered: {filtered_symbols}")
            
            # Step 3: DeepSeek V3 deep analysis
            deepseek_v3_analyses = await self.ai_analyzer.deepseek_v3_analysis(filtered_data)
            logger.info(f"🧠 DeepSeek V3 analyzed: {len(deepseek_v3_analyses)} stocks")
            
            # Step 4: DeepSeek R1 final decisions
            final_decisions = await self.ai_analyzer.deepseek_r1_decision(
                deepseek_v3_analyses,
                filtered_data
            )
            logger.info(f"🎯 DeepSeek R1 decisions: {len(final_decisions)}")
            
            # Step 5: Send alerts
            if final_decisions:
                await self.send_ai_alerts(final_decisions)
            
        except Exception as e:
            logger.error(f"Error in AI scan: {e}")
    
    async def send_ai_alerts(self, decisions):
        """AI decisions को Telegram पर भेजतो"""
        try:
            msg = "🤖 *AI TRADING ALERTS*\n"
            msg += f"⏰ {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n"
            
            for decision in decisions:
                symbol = decision.get("sym", "")
                action = decision.get("action", "HOLD")
                entry = decision.get("entry", 0)
                target = decision.get("target", 0)
                sl = decision.get("stop_loss", 0)
                conf = decision.get("confidence", 0)
                reason = decision.get("reason", "")
                
                # Action emoji
                emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
                
                msg += f"{emoji} *{symbol}* - {action}\n"
                msg += f"💰 Entry: ₹{entry:.2f}\n"
                msg += f"🎯 Target: ₹{target:.2f}\n"
                msg += f"🛑 SL: ₹{sl:.2f}\n"
                msg += f"📊 Confidence: {conf}%\n"
                msg += f"💡 {reason}\n\n"
            
            msg += "━━━━━━━━━━━━━━━━\n"
            msg += "_Powered by 3-Layer AI System_\n"
            msg += "🧠 Cerebras → Hyperbolic → DeepSeek R1"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            
            logger.info("✅ AI alerts sent")
            
        except Exception as e:
            logger.error(f"Error sending AI alerts: {e}")
    
    async def run(self):
        """Main loop - हर 1 minute AI scan"""
        logger.info("🚀 Bot started! Loading security IDs...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("Failed to load security IDs. Exiting...")
            return
        
        await self.send_startup_message()
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*50}")
                logger.info(f"🤖 AI Scan Cycle: {timestamp}")
                logger.info(f"{'='*50}")
                
                # AI scanning आणि alerts
                await self.ai_scan_and_alert()
                
                logger.info("✅ AI scan completed!")
                logger.info("⏳ Waiting 1 minute for next scan...\n")
                
                # 1 minute wait
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup message"""
        try:
            msg = "🤖 *AI Option Chain Bot Started!*\n\n"
            msg += f"📊 Tracking {len(self.security_id_map)} stocks/indices\n"
            msg += "⏱️ AI Scans every 1 minute\n\n"
            msg += "🧠 *2-Layer AI System:*\n"
            msg += "1️⃣ Cerebras Llama 3.3 70B - Quick Filter\n"
            msg += "2️⃣ DeepSeek V3 - Deep Analysis\n"
            msg += "3️⃣ DeepSeek R1 - Final Decisions\n\n"
            msg += "✅ Powered by DhanHQ API v2\n"
            msg += "🚂 Deployed on Railway.app"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("Startup message sent")
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")


# ========================
# BOT RUN
# ========================
if __name__ == "__main__":
    try:
        # Environment variables check with detailed logging
        missing_vars = []
        
        if not TELEGRAM_BOT_TOKEN:
            missing_vars.append("TELEGRAM_BOT_TOKEN")
        if not TELEGRAM_CHAT_ID:
            missing_vars.append("TELEGRAM_CHAT_ID")
        if not DHAN_CLIENT_ID:
            missing_vars.append("DHAN_CLIENT_ID")
        if not DHAN_ACCESS_TOKEN:
            missing_vars.append("DHAN_ACCESS_TOKEN")
        if not CEREBRAS_API_KEY:
            missing_vars.append("CEREBRAS_API_KEY")
        if not DEEPSEEK_API_KEY:
            missing_vars.append("DEEPSEEK_API_KEY")
        
        if missing_vars:
            logger.error("❌ Missing environment variables:")
            for var in missing_vars:
                logger.error(f"  - {var}")
            logger.error("\n🔧 Fix: Add these variables in Railway/Render dashboard")
            exit(1)
        
        logger.info("✅ All environment variables loaded!")
        logger.info(f"📱 Telegram Chat ID: {TELEGRAM_CHAT_ID}")
        logger.info(f"🔑 Dhan Client ID: {DHAN_CLIENT_ID[:10]}...")
        
        bot = DhanOptionChainBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)
