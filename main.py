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
from zoneinfo import ZoneInfo

# ========================
# TIMEZONE FIX - IST (UTC+5:30)
# ========================
IST = ZoneInfo('Asia/Kolkata')

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
# DATA COMPRESSION HELPER
# ========================
def compress_candles(candles, last_n=30):
    """Only last N candles + summary statistics"""
    if not candles or len(candles) == 0:
        return {"error": "No candles"}
    
    recent = candles[-last_n:] if len(candles) > last_n else candles
    
    all_highs = [c['high'] for c in candles]
    all_lows = [c['low'] for c in candles]
    all_closes = [c['close'] for c in candles]
    all_volumes = [c['volume'] for c in candles]
    
    return {
        "recent_candles": recent,
        "total_candles": len(candles),
        "period_high": max(all_highs),
        "period_low": min(all_lows),
        "avg_close": sum(all_closes) / len(all_closes),
        "total_volume": sum(all_volumes),
        "first_close": candles[0]['close'],
        "last_close": candles[-1]['close'],
        "change_pct": ((candles[-1]['close'] - candles[0]['close']) / candles[0]['close'] * 100) if candles[0]['close'] != 0 else 0
    }

def compress_option_chain(oc_data, spot_price):
    """Extract only key option chain metrics"""
    if not oc_data or not oc_data.get('oc'):
        return {"error": "No option chain data"}
    
    oc = oc_data.get('oc', {})
    strikes = sorted([float(s) for s in oc.keys()])
    
    if not strikes or spot_price == 0:
        return {"error": "Invalid strikes or spot price"}
    
    # ATM strike
    atm = min(strikes, key=lambda x: abs(x - spot_price))
    atm_idx = strikes.index(atm)
    
    # ATM + 5 OTM strikes each side
    start = max(0, atm_idx - 5)
    end = min(len(strikes), atm_idx + 6)
    key_strikes = strikes[start:end]
    
    # Extract key data
    chain_summary = []
    total_ce_oi = 0
    total_pe_oi = 0
    
    for strike in key_strikes:
        strike_key = f"{strike:.6f}"
        data = oc.get(strike_key, {})
        
        ce = data.get('ce', {})
        pe = data.get('pe', {})
        
        ce_oi = ce.get('oi', 0)
        pe_oi = pe.get('oi', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        
        chain_summary.append({
            'strike': strike,
            'ce_ltp': ce.get('last_price', 0),
            'ce_oi': ce_oi,
            'ce_vol': ce.get('volume', 0),
            'pe_ltp': pe.get('last_price', 0),
            'pe_oi': pe_oi,
            'pe_vol': pe.get('volume', 0),
            'is_atm': strike == atm
        })
    
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    return {
        "spot_price": spot_price,
        "atm_strike": atm,
        "pcr": round(pcr, 2),
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "key_strikes": chain_summary
    }


# ========================
# LAYER 1: GROQ FAST FILTER (FREE)
# ========================
class GroqFilter:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        logger.info("✅ Layer 1: Groq (Llama 3.3) - FREE")
    
    def quick_filter(self, symbol, candles_summary, oc_summary):
        """Fast filtering using COMPRESSED data"""
        try:
            prompt = f"""Quick F&O pre-filter for {symbol}.

CANDLESTICK SUMMARY (Last 30 of {candles_summary.get('total_candles', 0)} candles):
- Period High/Low: ₹{candles_summary.get('period_high', 0):.2f} / ₹{candles_summary.get('period_low', 0):.2f}
- First → Last Close: ₹{candles_summary.get('first_close', 0):.2f} → ₹{candles_summary.get('last_close', 0):.2f}
- Change: {candles_summary.get('change_pct', 0):.2f}%
- Total Volume: {candles_summary.get('total_volume', 0):,.0f}

Recent 30 Candles:
{json.dumps(candles_summary.get('recent_candles', []), indent=1)}

OPTION CHAIN:
- Spot: ₹{oc_summary.get('spot_price', 0):.2f}
- ATM: ₹{oc_summary.get('atm_strike', 0):.0f}
- PCR: {oc_summary.get('pcr', 0):.2f}
- CE OI: {oc_summary.get('total_ce_oi', 0):,.0f} | PE OI: {oc_summary.get('total_pe_oi', 0):,.0f}

Key Strikes:
{json.dumps(oc_summary.get('key_strikes', []), indent=1)}

Score 0-10 based on momentum, volume, PCR, setup quality.

OUTPUT:
SCORE: [0-10]
REASON: [One line]"""
            
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
    
    async def detailed_analysis(self, symbol, candles_summary, oc_summary, groq_reason):
        """Detailed analysis with COMPRESSED data"""
        try:
            spot = oc_summary.get('spot_price', 0)
            
            prompt = f"""Expert F&O analysis for {symbol}.

SPOT: ₹{spot:,.2f}
TIME: {datetime.now(IST).strftime('%d-%m-%Y %H:%M IST')}

CANDLES ({candles_summary.get('total_candles', 0)} total, showing last 30):
- Range: ₹{candles_summary.get('period_low', 0):.2f} - ₹{candles_summary.get('period_high', 0):.2f}
- Movement: {candles_summary.get('change_pct', 0):.2f}%
- Recent: {json.dumps(candles_summary.get('recent_candles', [])[-10:], indent=1)}

OPTION CHAIN:
- ATM: ₹{oc_summary.get('atm_strike', 0):.0f} | PCR: {oc_summary.get('pcr', 0):.2f}
- Strikes: {json.dumps(oc_summary.get('key_strikes', []), indent=1)}

GROQ: {groq_reason}

Analyze price action, option chain, and provide trade setup.

OUTPUT:

🎯 DECISION: [YES/NO/WAIT]

IF YES:
📊 DIRECTION: [BULLISH/BEARISH]
Entry: ₹[price]
T1: ₹[price]
T2: ₹[price]
SL: ₹[price] ([X]%)
R:R: [X:Y]
CONFIDENCE: [X]%
REASONS:
- [Key reason 1]
- [Key reason 2]

IF NO/WAIT:
❌ REASON: [Why]"""
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            analysis = response.choices[0].message.content
            
            # Cost
            usage = response.usage
            cost_input = (usage.prompt_tokens / 1_000_000) * 0.27
            cost_output = (usage.completion_tokens / 1_000_000) * 1.10
            cost_usd = cost_input + cost_output
            cost_inr = cost_usd * 83
            
            logger.info(f"  💎 DeepSeek V3: {symbol} | Cost: ₹{cost_inr:.4f}")
            
            # Decision
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
            prompt = f"""Final trade decision for {symbol} (Spot: ₹{spot_price:,.2f}).

V3 Analysis:
{deepseek_v3_analysis}

Deep reasoning:
1. Clear entry trigger?
2. Safe SL?
3. R:R > 1.5?
4. Hidden risks?
5. OC confirms?
6. High probability?

OUTPUT:

🧠 REASONING:
[Step-by-step thinking]

🎯 FINAL: [TRADE/SKIP]

IF TRADE:
✅ VALIDATED: [Entry, targets, SL]
⚠️ RISKS:
- [Risk 1]
- [Risk 2]

IF SKIP:
❌ REASON: [Why]

Be conservative."""
            
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
            
            # Decision
            final_decision = "SKIP"
            if "FINAL: TRADE" in decision_text.upper() or "FINAL DECISION: TRADE" in decision_text.upper():
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
        
        # AI layers
        self.groq = GroqFilter(GROQ_API_KEY)
        self.deepseek_v3 = DeepSeekV3Analyzer(DEEPSEEK_API_KEY)
        self.deepseek_r1 = DeepSeekR1FinalDecision(DEEPSEEK_API_KEY)
        
        self.total_cost = 0
        
        logger.info("🤖 3-Layer AI Bot initialized")
    
    def get_ist_time(self):
        return datetime.now(IST)
    
    async def load_security_ids(self):
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
        try:
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
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
                    
                    result = candles[-100:] if len(candles) > 100 else candles
                    logger.info(f"  📊 Fetched {len(result)} candles")
                    return result
            
            return None
        except Exception as e:
            logger.error(f"Candles error: {e}")
            return None
    
    def get_option_chain(self, security_id, segment):
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
                logger.warning(f"Expiry API failed: {expiry_response.status_code}")
                return None
            
            expiry_data = expiry_response.json()
            if not expiry_data.get('data'):
                logger.warning(f"No expiry data: {expiry_data}")
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
                data = oc_response.json()
                # Debug: Log spot price
                spot = data.get('last_price', 0)
                logger.info(f"  OC: Spot=₹{spot:.2f}, Status={data.get('status', 'unknown')}")
                return data
            else:
                logger.warning(f"OC API failed: {oc_response.status_code}")
            
            return None
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None
    
    async def analyze_symbol(self, symbol):
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
                logger.warning(f"  ⚠️ Insufficient candles ({len(candles) if candles else 0})")
                return
            
            option_chain = self.get_option_chain(info['security_id'], info['segment'])
            if not option_chain:
                logger.warning(f"  ⚠️ No option chain")
                return
            
            spot_price = option_chain.get('last_price', 0)
            
            # Check spot price validity
            if spot_price == 0 or spot_price is None:
                logger.warning(f"  ⚠️ Invalid spot price (Market closed or data unavailable)")
                return
            
            # COMPRESS DATA
            candles_summary = compress_candles(candles, last_n=30)
            oc_summary = compress_option_chain(option_chain, spot_price)
            
            logger.info(f"  📊 Data: {len(candles)} candles → 30, Spot: ₹{spot_price:,.2f}, PCR: {oc_summary.get('pcr', 0)}")
            
            # LAYER 1: GROQ
            groq_result = self.groq.quick_filter(symbol, candles_summary, oc_summary)
            
            if not groq_result['passed']:
                logger.info(f"  ⏭️ Filtered (Score: {groq_result['score']}/10)")
                return
            
            logger.info(f"  ✅ Passed Groq")
            
            # LAYER 2: DeepSeek V3
            v3_result = await self.deepseek_v3.detailed_analysis(
                symbol, candles_summary, oc_summary, groq_result['reason']
            )
            
            if not v3_result or v3_result['decision'] != "YES":
                logger.info(f"  ⏭️ V3: {v3_result['decision'] if v3_result else 'ERROR'}")
                if v3_result:
                    self.total_cost += v3_result['cost']
                return
            
            logger.info(f"  ✅ V3: Trade recommended")
            self.total_cost += v3_result['cost']
            
            # LAYER 3: DeepSeek R1
            r1_result = await self.deepseek_r1.final_decision(
                symbol, spot_price, v3_result['analysis']
            )
            
            if not r1_result or r1_result['decision'] != "TRADE":
                logger.info(f"  ⏭️ R1: {r1_result['decision'] if r1_result else 'ERROR'}")
                return
            
            logger.info(f"  🎯 R1: TRADE CONFIRMED!")
            
            await self.send_trade_alert(symbol, spot_price, v3_result, r1_result)
            
        except Exception as e:
            logger.error(f"Analysis error {symbol}: {e}")
    
    async def send_trade_alert(self, symbol, spot_price, v3_result, r1_result):
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
        try:
            msg = "🤖 *3-LAYER AI BOT ACTIVE*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            msg += f"⏰ Time: {self.get_ist_time().strftime('%d-%m-%Y %H:%M IST')}\n"
            msg += f"📊 Symbols: 10 (8 stocks + 2 indices)\n"
            msg += f"⏱️  Interval: {SCAN_INTERVAL_MINUTES} minutes\n"
            msg += f"📈 Data: Last 30 candles (compressed)\n\n"
            
            msg += "🔥 *AI PIPELINE*:\n\n"
            msg += "🟢 Layer 1: GROQ (Compressed) - FREE\n"
            msg += "💎 Layer 2: DeepSeek V3 - $0.004/scan\n"
            msg += "🧠 Layer 3: DeepSeek R1 - FREE\n\n"
            
            msg += "🎯 Status: Monitoring...\n"
            msg += f"📅 Market: {MARKET_OPEN}-{MARKET_CLOSE} IST (Mon-Fri)"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup sent")
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    def is_market_hours(self):
        now = self.get_ist_time()
        current_time = now.time()
        is_weekday = now.weekday() < 5
        
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
        
        # Wait for market to open if starting before market hours
        while not self.is_market_hours():
            now = self.get_ist_time()
            logger.info(f"\n⏸️  Waiting for market to open...")
            logger.info(f"Current: {now.strftime('%H:%M IST')} | Market: {MARKET_OPEN}-{MARKET_CLOSE} IST")
            logger.info(f"Next check in 10 minutes...\n")
            await asyncio.sleep(600)  # 10 minutes
        
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
