import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import csv
import json
from groq import Groq
from openai import OpenAI
import pytz

# ========================
# TIMEZONE FIX - IST (UTC+5:30)
# ========================
IST = pytz.timezone('Asia/Kolkata')

# Logging
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Dhan API
DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# ========================
# TOP 8 F&O STOCKS + 2 INDICES
# ========================
SYMBOLS = {
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "TCS": {"symbol": "TCS", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ"},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
}

# Market Hours (IST)
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
SCAN_INTERVAL_MINUTES = 5

# ========================
# LAYER 1: GROQ FAST FILTER (FREE)
# ========================
class GroqFilter:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        logger.info("✅ Layer 1: Groq (Llama 3.3) - FREE")
    
    def quick_filter(self, symbol, candles_json, option_chain_json, spot_price):
        """Fast filtering using Groq (FREE)"""
        try:
            prompt = f"""You are a quick pre-filter for Indian F&O trading.

SYMBOL: {symbol}
SPOT PRICE: ₹{spot_price:,.2f}

CANDLESTICK DATA (Last 100 candles, 5-min):
{candles_json}

OPTION CHAIN DATA:
{option_chain_json}

YOUR TASK:
Score this setup 0-10 based on:
1. Price momentum (last 20 candles)
2. Volume confirmation
3. Option chain sentiment
4. Trend or breakout setup
5. Risk-reward potential

OUTPUT (strict format):
SCORE: [0-10]
REASON: [One line]

Example:
SCORE: 8
REASON: Strong bullish with volume

Analyze {symbol}:"""
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content
            
            # Extract score
            score = 0
            reason = "No clear setup"
            
            lines = result.strip().split('\n')
            for line in lines:
                if 'SCORE:' in line.upper():
                    try:
                        score = int(''.join(filter(str.isdigit, line)))
                    except:
                        score = 0
                elif 'REASON:' in line.upper():
                    reason = line.split(':', 1)[1].strip()
            
            logger.info(f"  ⚡ Groq: {symbol} → {score}/10 | {reason}")
            
            return {
                'score': score,
                'reason': reason,
                'passed': score >= 6
            }
            
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return {'score': 0, 'reason': 'Error', 'passed': False}


# ========================
# LAYER 2: DEEPSEEK V3 ANALYSIS
# ========================
class DeepSeekV3Analyzer:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        logger.info("✅ Layer 2: DeepSeek V3 - $0.004/scan")
    
    async def detailed_analysis(self, symbol, candles_json, option_chain_json, spot_price, groq_reason):
        """Detailed analysis with DeepSeek V3"""
        try:
            prompt = f"""You are an expert F&O trader.

SYMBOL: {symbol}
SPOT: ₹{spot_price:,.2f}
TIME: {datetime.now(IST).strftime('%d-%m-%Y %H:%M IST')}

CANDLESTICKS (100 candles, 5-min):
{candles_json}

OPTION CHAIN:
{option_chain_json}

GROQ PRE-FILTER: {groq_reason}

ANALYZE:
1. Price Action (last 100 candles, trends, levels)
2. Option Chain (PCR, OI strikes, support/resistance)
3. Trade Setup (direction, entry, targets, SL)

OUTPUT FORMAT:

🎯 DECISION: [YES/NO/WAIT]

IF YES:
📊 DIRECTION: [BULLISH/BEARISH]
Entry: ₹[price]
Entry Condition: [Trigger]
T1: ₹[price]
T2: ₹[price]
SL: ₹[price] ([X]% risk)
RISK:REWARD: [X:Y]
CONFIDENCE: [X]%
KEY REASONS:
- [Reason 1]
- [Reason 2]

IF NO:
❌ REASON: [Why not]

Analyze {symbol}:"""
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            
            # Cost calculation (DeepSeek V3)
            usage = response.usage
            cost_input = (usage.prompt_tokens / 1_000_000) * 0.27
            cost_output = (usage.completion_tokens / 1_000_000) * 1.10
            cost_usd = cost_input + cost_output
            cost_inr = cost_usd * 83
            
            logger.info(f"  💎 DeepSeek V3: {symbol} | Cost: ₹{cost_inr:.4f}")
            
            # Check decision
            decision = "NO"
            if "DECISION: YES" in analysis.upper():
                decision = "YES"
            elif "DECISION: WAIT" in analysis.upper():
                decision = "WAIT"
            
            return {
                'decision': decision,
                'analysis': analysis,
                'cost': cost_inr
            }
            
        except Exception as e:
            logger.error(f"DeepSeek V3 error: {e}")
            return None


# ========================
# LAYER 3: DEEPSEEK R1 FINAL DECISION
# ========================
class DeepSeekR1FinalDecision:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        logger.info("✅ Layer 3: DeepSeek R1 - FREE")
    
    async def final_decision(self, symbol, spot_price, deepseek_v3_analysis):
        """Final decision with DeepSeek R1"""
        try:
            prompt = f"""Final decision maker for F&O trades.

SYMBOL: {symbol}
SPOT: ₹{spot_price:,.2f}

DeepSeek V3 Analysis:
{deepseek_v3_analysis}

TASK - DEEP REASONING:
1. Is entry trigger clear?
2. Is SL safe and structure-based?
3. Is risk:reward favorable (>1.5)?
4. Any hidden risks?
5. Does option chain confirm?
6. High-probability setup?

Make FINAL DECISION: TRADE or SKIP

OUTPUT:

🧠 REASONING:
[Step-by-step thinking]

🎯 FINAL DECISION: [TRADE/SKIP]

IF TRADE:
✅ SETUP VALIDATED: [Confirm entry, targets, SL]
⚠️ RISK FACTORS:
- [Risk 1]
- [Risk 2]

IF SKIP:
❌ REASON: [Why skip]

Be conservative. Only TRADE if truly confident.

Analyze {symbol}:"""
            
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            reasoning = ""
            decision_text = ""
            
            message = response.choices[0].message
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning = message.reasoning_content
            decision_text = message.content
            
            logger.info(f"  🧠 DeepSeek R1: {symbol} | FREE")
            
            # Check final decision
            final_decision = "SKIP"
            if "FINAL DECISION: TRADE" in decision_text.upper():
                final_decision = "TRADE"
            
            return {
                'decision': final_decision,
                'reasoning': reasoning,
                'analysis': decision_text
            }
            
        except Exception as e:
            logger.error(f"DeepSeek R1 error: {e}")
            return None


# ========================
# MAIN BOT
# ========================
class TradingBot:
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
        
        # Initialize AI layers
        self.groq = GroqFilter(GROQ_API_KEY)
        self.deepseek_v3 = DeepSeekV3Analyzer(DEEPSEEK_API_KEY)
        self.deepseek_r1 = DeepSeekR1FinalDecision(DEEPSEEK_API_KEY)
        
        self.total_cost = 0
        
        logger.info("🤖 3-Layer AI Bot initialized")
    
    def get_ist_time(self):
        """Get current time in IST"""
        return datetime.now(IST)
    
    async def load_security_ids(self):
        """Load security IDs from Dhan"""
        try:
            logger.info("Loading IDs...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code == 200:
                csv_data = response.text.split('\n')
                
                for symbol, info in SYMBOLS.items():
                    reader = csv.DictReader(csv_data)
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
                                        logger.info(f"✅ {symbol}")
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
                                        logger.info(f"✅ {symbol}")
                                        break
                        except:
                            continue
                
                logger.info(f"✅ Loaded {len(self.security_id_map)} symbols\n")
                return True
            return False
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False
    
    def get_candles(self, security_id, segment, symbol):
        """Fetch last 100 candles (5-min)"""
        try:
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
            to_date = self.get_ist_time()
            from_date = to_date - timedelta(days=5)
            
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
                
                if 'open' in data and len(data['open']) > 0:
                    candles = []
                    for i in range(len(data['open'])):
                        candles.append({
                            'open': round(data['open'][i], 2),
                            'high': round(data['high'][i], 2),
                            'low': round(data['low'][i], 2),
                            'close': round(data['close'][i], 2),
                            'volume': int(data['volume'][i])
                        })
                    
                    logger.info(f"  📊 Fetched {len(candles)} candles")
                    return candles[-100:] if len(candles) > 100 else candles
            
            return None
        except Exception as e:
            logger.error(f"Candles error: {e}")
            return None
    
    def get_option_chain(self, security_id, segment):
        """Get option chain data"""
        try:
            expiry_payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment
            }
            
            expiry_response = requests.post(
                DHAN_EXPIRY_LIST_URL,
                json=expiry_payload,
                headers=self.headers,
                timeout=10
            )
            
            if expiry_response.status_code != 200:
                return None
            
            expiry_data = expiry_response.json()
            if not expiry_data.get('data'):
                return None
            
            expiry = expiry_data['data'][0]
            
            oc_payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment,
                "Expiry": expiry
            }
            
            oc_response = requests.post(
                DHAN_OPTION_CHAIN_URL,
                json=oc_payload,
                headers=self.headers,
                timeout=15
            )
            
            if oc_response.status_code == 200:
                return oc_response.json()
            
            return None
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None
    
    async def analyze_symbol(self, symbol):
        """3-Layer AI Pipeline"""
        try:
            if symbol not in self.security_id_map:
                return
            
            info = self.security_id_map[symbol]
            logger.info(f"\n{'─'*50}")
            logger.info(f"📊 {symbol}")
            logger.info(f"{'─'*50}")
            
            # Get data
            candles = self.get_candles(info['security_id'], info['segment'], symbol)
            if not candles or len(candles) < 20:
                logger.warning(f"  ⚠️ Insufficient candles")
                return
            
            option_chain = self.get_option_chain(info['security_id'], info['segment'])
            if not option_chain:
                logger.warning(f"  ⚠️ No option chain")
                return
            
            spot_price = option_chain.get('last_price', 0)
            
            # Convert to JSON
            candles_json = json.dumps(candles, indent=2)
            oc_json = json.dumps(option_chain, indent=2)
            
            logger.info(f"  📊 Data: {len(candles)} candles, Spot: ₹{spot_price:,.2f}")
            
            # ==================
            # LAYER 1: GROQ
            # ==================
            groq_result = self.groq.quick_filter(symbol, candles_json, oc_json, spot_price)
            
            if not groq_result['passed']:
                logger.info(f"  ⏭️ Filtered (Score: {groq_result['score']}/10)")
                return
            
            logger.info(f"  ✅ Passed Groq")
            
            # ==================
            # LAYER 2: DeepSeek V3
            # ==================
            v3_result = await self.deepseek_v3.detailed_analysis(
                symbol, candles_json, oc_json, spot_price, groq_result['reason']
            )
            
            if not v3_result or v3_result['decision'] != "YES":
                logger.info(f"  ⏭️ V3: {v3_result['decision'] if v3_result else 'ERROR'}")
                if v3_result:
                    self.total_cost += v3_result['cost']
                return
            
            logger.info(f"  ✅ V3: Trade recommended")
            self.total_cost += v3_result['cost']
            
            # ==================
            # LAYER 3: DeepSeek R1
            # ==================
            r1_result = await self.deepseek_r1.final_decision(
                symbol, spot_price, v3_result['analysis']
            )
            
            if not r1_result or r1_result['decision'] != "TRADE":
                logger.info(f"  ⏭️ R1: {r1_result['decision'] if r1_result else 'ERROR'}")
                return
            
            logger.info(f"  🎯 R1: TRADE CONFIRMED!")
            
            # Send alert
            await self.send_trade_alert(symbol, spot_price, v3_result, r1_result)
            
        except Exception as e:
            logger.error(f"Analysis error {symbol}: {e}")
    
    async def send_trade_alert(self, symbol, spot_price, v3_result, r1_result):
        """Send trade alert to Telegram"""
        try:
            msg = f"🚨 *TRADE READY* 🚨\n"
            msg += f"{'═'*40}\n\n"
            
            msg += f"📊 *{symbol}*\n"
            msg += f"💰 Spot: ₹{spot_price:,.2f}\n"
            msg += f"⏰ {self.get_ist_time().strftime('%d-%m-%Y %H:%M IST')}\n\n"
            
            msg += f"{'═'*40}\n"
            msg += f"💎 *V3 ANALYSIS*\n"
            msg += f"{'═'*40}\n\n"
            msg += f"```\n{v3_result['analysis']}\n```\n\n"
            
            msg += f"{'═'*40}\n"
            msg += f"🧠 *R1 REASONING*\n"
            msg += f"{'═'*40}\n\n"
            msg += f"```\n{r1_result['analysis']}\n```\n\n"
            
            msg += f"{'═'*40}\n"
            msg += f"💰 Cost: ₹{v3_result['cost']:.4f} | Total: ₹{self.total_cost:.2f}"
            
            # Split if too long
            if len(msg) > 4000:
                parts = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
                for idx, part in enumerate(parts, 1):
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=f"[Part {idx}/{len(parts)}]\n\n{part}",
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(1)
            else:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg,
                    parse_mode='Markdown'
                )
            
            logger.info(f"  ✅ Alert sent")
        except Exception as e:
            logger.error(f"Alert error: {e}")
    
    async def send_startup_message(self):
        """Send startup message"""
        try:
            msg = "🤖 *3-LAYER AI BOT ACTIVE*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            msg += f"⏰ Time: {self.get_ist_time().strftime('%d-%m-%Y %H:%M IST')}\n"
            msg += f"📊 Symbols: 10 (8 stocks + 2 indices)\n"
            msg += f"⏱️  Interval: {SCAN_INTERVAL_MINUTES} minutes\n"
            msg += f"📈 Candles: 100 (5-min) + Option chain\n\n"
            
            msg += "🔥 *AI PIPELINE*:\n\n"
            msg += "🟢 Layer 1: GROQ (Llama 3.3) - FREE\n"
            msg += "💎 Layer 2: DeepSeek V3 - $0.004/scan\n"
            msg += "🧠 Layer 3: DeepSeek R1 - FREE\n\n"
            
            msg += "🎯 Result: 2 ready-to-trade signals ✅"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup sent")
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    def is_market_hours(self):
        """Check if market is open (IST timezone)"""
        now = self.get_ist_time()
        current_time = now.time()
        is_weekday = now.weekday() < 5  # Mon-Fri
        
        market_open_time = datetime.strptime(MARKET_OPEN, "%H:%M").time()
        market_close_time = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
        
        market_open = current_time >= market_open_time
        market_close = current_time <= market_close_time
        is_market_hours = is_weekday and market_open and market_close
        
        return is_market_hours
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting 3-Layer AI Bot...\n")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load IDs")
            return
        
        await self.send_startup_message()
        
        symbols = list(self.security_id_map.keys())
        
        while self.running:
            try:
                now = self.get_ist_time()
                
                if not self.is_market_hours():
                    logger.info(f"\n⏸️  Market closed")
                    logger.info(f"Current: {now.strftime('%H:%M IST')} | Market: {MARKET_OPEN}-{MARKET_CLOSE} IST")
                    logger.info(f"Next check in 30 minutes...\n")
                    await asyncio.sleep(1800)
                    continue
                
                logger.info(f"\n{'═'*60}")
                logger.info(f"🔄 SCAN: {now.strftime('%d-%m-%Y %H:%M:%S IST')}")
                logger.info(f"{'═'*60}")
                
                for symbol in symbols:
                    await self.analyze_symbol(symbol)
                    await asyncio.sleep(2)
                
                logger.info(f"\n{'═'*60}")
                logger.info(f"✅ Cycle complete | Total cost: ₹{self.total_cost:.2f}")
                logger.info(f"⏳ Next scan in {SCAN_INTERVAL_MINUTES} minutes...")
                logger.info(f"{'═'*60}\n")
                
                await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(60)


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    try:
        required = {
            'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
            'DHAN_CLIENT_ID': DHAN_CLIENT_ID,
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN,
            'GROQ_API_KEY': GROQ_API_KEY,
            'DEEPSEEK_API_KEY': DEEPSEEK_API_KEY
        }
        
        missing = [k for k, v in required.items() if not v]
        
        if missing:
            logger.error(f"❌ Missing: {', '.join(missing)}")
            exit(1)
        
        logger.info("✅ All variables OK\n")
        
        bot = TradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
