"""
🤖 NIFTY 50 TRADING BOT v10.4 - WEEKLY EXPIRY FIX
✅ Weekly Expiry Detection (Every Tuesday)
✅ Correct Strike Price Range
✅ ATM Detection Fix
"""

import asyncio
import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import traceback
import pytz
import redis
from dataclasses import dataclass
import html
import re

from telegram import Bot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Bot Configuration - NIFTY 50 v10.4"""
    
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    DHAN_API_BASE = "https://api.dhan.co"
    DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
    DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
    DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
    
    SCAN_INTERVAL = 300  # 5 minutes
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 86400
    
    CONFIDENCE_THRESHOLD = 70
    MIN_OI_DIVERGENCE_PCT = 3.0
    PCR_BULLISH_MIN = 1.1
    PCR_BEARISH_MAX = 0.9
    MIN_TOTAL_OI = 100000
    SKIP_OPENING_MINUTES = 10
    SKIP_CLOSING_MINUTES = 20
    
    LOOKBACK_DAYS = 15
    ATM_STRIKES_RANGE = 10  # ATM ± 10 = 21 strikes
    MIN_CANDLES_REQUIRED = 50
    
    # NIFTY 50
    INDEX_NAME = "NIFTY 50"
    INDEX_SECURITY_ID = 13  # Spot
    INDEX_SEGMENT = "IDX_I"
    FNO_SECURITY_ID = 13  # F&O
    FNO_SEGMENT = "NSE_FNO"


@dataclass
class OIData:
    strike: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_oi_change: int = 0
    pe_oi_change: int = 0
    ce_iv: float = 0.0
    pe_iv: float = 0.0
    pcr_at_strike: float = 0.0


@dataclass
class AggregateOIAnalysis:
    total_ce_oi: int
    total_pe_oi: int
    total_ce_volume: int
    total_pe_volume: int
    total_ce_oi_change: int
    total_pe_oi_change: int
    ce_oi_change_pct: float
    pe_oi_change_pct: float
    pcr: float
    overall_sentiment: str
    max_pain: float
    max_pain_distance: float


@dataclass
class AdvancedAnalysis:
    opportunity: str
    confidence: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    recommended_strike: int
    pattern_signal: str
    oi_flow_signal: str
    market_structure: str
    support_levels: List[float]
    resistance_levels: List[float]


class RedisCache:
    def __init__(self):
        try:
            logger.info("Connecting to Redis...")
            self.redis_client = redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected!")
        except Exception as e:
            logger.error(f"❌ Redis failed: {e}")
            self.redis_client = None
    
    def store_option_chain(self, symbol: str, oi_data: List[OIData], spot_price: float):
        try:
            if not self.redis_client:
                return False
            
            key = f"oi_data:{symbol}"
            value = json.dumps({
                'spot_price': spot_price,
                'strikes': [
                    {
                        'strike': oi.strike,
                        'ce_oi': oi.ce_oi,
                        'pe_oi': oi.pe_oi,
                        'ce_volume': oi.ce_volume,
                        'pe_volume': oi.pe_volume
                    }
                    for oi in oi_data
                ],
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
            })
            
            self.redis_client.setex(key, Config.REDIS_EXPIRY, value)
            return True
        except Exception as e:
            logger.error(f"Redis store error: {e}")
            return False
    
    def get_oi_comparison(self, symbol: str, current_oi: List[OIData], current_price: float) -> Dict:
        try:
            if not self.redis_client:
                return {'change': 'NO_CACHE', 'aggregate_analysis': None}
            
            key = f"oi_data:{symbol}"
            cached = self.redis_client.get(key)
            
            if not cached:
                return {'change': 'FIRST_SCAN', 'aggregate_analysis': None}
            
            old_data = json.loads(cached)
            old_strikes = {s['strike']: s for s in old_data['strikes']}
            
            total_ce_oi_old = sum(s['ce_oi'] for s in old_data['strikes'])
            total_pe_oi_old = sum(s['pe_oi'] for s in old_data['strikes'])
            
            total_ce_oi_new = sum(oi.ce_oi for oi in current_oi)
            total_pe_oi_new = sum(oi.pe_oi for oi in current_oi)
            
            ce_oi_change = total_ce_oi_new - total_ce_oi_old
            pe_oi_change = total_pe_oi_new - total_pe_oi_old
            
            ce_oi_change_pct = (ce_oi_change / total_ce_oi_old * 100) if total_ce_oi_old > 0 else 0
            pe_oi_change_pct = (pe_oi_change / total_pe_oi_old * 100) if total_pe_oi_old > 0 else 0
            
            pcr = total_pe_oi_new / total_ce_oi_new if total_ce_oi_new > 0 else 0
            
            sentiment = "NEUTRAL"
            if pe_oi_change_pct > 3 and pe_oi_change_pct > ce_oi_change_pct:
                sentiment = "BULLISH"
            elif ce_oi_change_pct > 3 and ce_oi_change_pct > pe_oi_change_pct:
                sentiment = "BEARISH"
            
            max_pain = self.calculate_max_pain(current_oi, current_price)
            max_pain_distance = ((current_price - max_pain) / current_price) * 100
            
            aggregate_analysis = AggregateOIAnalysis(
                total_ce_oi=total_ce_oi_new,
                total_pe_oi=total_pe_oi_new,
                total_ce_volume=sum(oi.ce_volume for oi in current_oi),
                total_pe_volume=sum(oi.pe_volume for oi in current_oi),
                total_ce_oi_change=ce_oi_change,
                total_pe_oi_change=pe_oi_change,
                ce_oi_change_pct=ce_oi_change_pct,
                pe_oi_change_pct=pe_oi_change_pct,
                pcr=pcr,
                overall_sentiment=sentiment,
                max_pain=max_pain,
                max_pain_distance=max_pain_distance
            )
            
            return {
                'change': 'UPDATED',
                'aggregate_analysis': aggregate_analysis
            }
            
        except Exception as e:
            logger.error(f"Redis comparison error: {e}")
            return {'change': 'ERROR', 'aggregate_analysis': None}
    
    def calculate_max_pain(self, oi_data: List[OIData], spot_price: float) -> float:
        try:
            max_pain_strike = spot_price
            min_total_loss = float('inf')
            
            for test_strike_data in oi_data:
                test_strike = test_strike_data.strike
                total_loss = 0
                
                for oi in oi_data:
                    if test_strike > oi.strike:
                        total_loss += oi.ce_oi * (test_strike - oi.strike)
                    
                    if test_strike < oi.strike:
                        total_loss += oi.pe_oi * (oi.strike - test_strike)
                
                if total_loss < min_total_loss:
                    min_total_loss = total_loss
                    max_pain_strike = test_strike
            
            return max_pain_strike
        except Exception as e:
            logger.error(f"Max Pain error: {e}")
            return spot_price


class DhanAPI:
    def __init__(self, redis_cache: RedisCache):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json'
        }
        self.redis = redis_cache
        logger.info(f"✅ DhanAPI initialized - NIFTY 50 v10.4")
    
    def get_nearest_weekly_expiry(self) -> Optional[str]:
        """
        ✅ FIXED: Get CURRENT or NEXT WEEKLY expiry (Tuesday)
        
        NIFTY 50 weekly expiry hamesha TUESDAY la asata.
        Agar aaj Tuesday ahe ani market open ahe tar aajchich expiry use karo,
        nahi tar pudchya Tuesday chi.
        """
        try:
            ist = pytz.timezone('Asia/Kolkata')
            today = datetime.now(ist).date()
            current_time = datetime.now(ist).time()
            market_close_time = datetime.strptime("15:30", "%H:%M").time()
            
            # Check if today is Tuesday and market is still open
            is_tuesday = today.weekday() == 1  # 0=Monday, 1=Tuesday
            is_market_open = current_time < market_close_time
            
            logger.info(f"📅 Today: {today} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][today.weekday()]})")
            logger.info(f"⏰ Time: {current_time.strftime('%H:%M')}")
            
            # Calculate next Tuesday
            days_until_tuesday = (1 - today.weekday()) % 7  # Days until next Tuesday
            
            if is_tuesday and is_market_open:
                # Use today's expiry if it's Tuesday and market open
                current_expiry = today
                logger.info(f"✅ Using TODAY's expiry (Tuesday, market open)")
            else:
                # Use next Tuesday
                if days_until_tuesday == 0:
                    days_until_tuesday = 7  # Next Tuesday if today is Tuesday but market closed
                current_expiry = today + timedelta(days=days_until_tuesday)
                logger.info(f"✅ Using NEXT Tuesday expiry ({days_until_tuesday} days)")
            
            expiry_str = current_expiry.strftime("%Y-%m-%d")
            
            # Verify with API
            payload = {
                "UnderlyingScrip": Config.FNO_SECURITY_ID,
                "UnderlyingSeg": Config.FNO_SEGMENT
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
                    logger.info(f"📋 Available expiries: {expiries[:5]}")
                    
                    # Check if our calculated expiry exists
                    if expiry_str in expiries:
                        logger.info(f"✅ Weekly Expiry CONFIRMED: {expiry_str}")
                        return expiry_str
                    else:
                        # Use nearest available expiry
                        logger.warning(f"⚠️ {expiry_str} not in list, using nearest")
                        for exp in expiries:
                            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                            if exp_date >= today:
                                logger.info(f"✅ Using nearest expiry: {exp}")
                                return exp
            
            logger.warning(f"⚠️ API failed, using calculated: {expiry_str}")
            return expiry_str
            
        except Exception as e:
            logger.error(f"❌ Expiry error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_multi_timeframe_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch MTF data"""
        try:
            logger.info(f"📊 Fetching chart data...")
            
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=Config.LOOKBACK_DAYS)
            
            payload = {
                "securityId": str(Config.INDEX_SECURITY_ID),
                "exchangeSegment": Config.INDEX_SEGMENT,
                "instrument": "INDEX",
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            response = requests.post(
                Config.DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Intraday failed: {response.status_code}")
                return None
            
            data = response.json()
            
            if 'timestamp' not in data or len(data.get('open', [])) == 0:
                logger.error(f"❌ No candle data")
                return None
            
            df_base = pd.DataFrame({
                'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data.get('volume', [0] * len(data['open']))
            })
            
            df_base = df_base.dropna()
            df_base.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Received {len(df_base)} candles")
            
            if len(df_base) < Config.MIN_CANDLES_REQUIRED:
                logger.warning(f"⚠️ Only {len(df_base)} candles")
                return None
            
            result = {}
            result['5m'] = df_base.copy()
            
            result['15m'] = df_base.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.info(f"✅ MTF: 5m={len(result['5m'])}, 15m={len(result['15m'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ MTF data error: {e}")
            return None
    
    def get_option_chain(self, expiry: str, spot_price: float) -> Optional[List[OIData]]:
        """
        ✅ FIXED: ATM detection with correct strike range
        """
        try:
            payload = {
                "UnderlyingScrip": Config.FNO_SECURITY_ID,
                "UnderlyingSeg": Config.FNO_SEGMENT,
                "Expiry": expiry
            }
            
            logger.info(f"📈 Fetching option chain...")
            
            response = requests.post(
                Config.DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Option chain failed: {response.status_code}")
                return None
            
            data = response.json()
            if not data.get('data'):
                logger.error("❌ No option chain data")
                return None
            
            oc_data = data['data'].get('oc', {})
            
            if not oc_data:
                logger.error("❌ No strikes")
                return None
            
            # Convert strikes to float and filter valid ones
            strikes = []
            for strike_str in oc_data.keys():
                try:
                    strike = float(strike_str)
                    # ✅ FILTER: Only keep strikes near spot price
                    # NIFTY 50 strikes are in 50 point intervals
                    # Keep strikes within ±2000 points of spot
                    if abs(strike - spot_price) <= 2000:
                        strikes.append(strike)
                except:
                    continue
            
            strikes = sorted(strikes)
            
            if not strikes:
                logger.error("❌ No valid strikes found near spot price")
                return None
            
            logger.info(f"📊 Total valid strikes: {len(strikes)}")
            logger.info(f"📊 Strike range: {min(strikes):.0f} to {max(strikes):.0f}")
            logger.info(f"💰 Spot: {spot_price:.2f}")
            
            # ✅ Find ATM (nearest strike to spot)
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            atm_idx = strikes.index(atm_strike)
            
            logger.info(f"🎯 ATM Strike: {atm_strike:.0f} (Distance: {abs(atm_strike - spot_price):.2f})")
            
            # ✅ Select ATM ± 10 strikes
            start_idx = max(0, atm_idx - Config.ATM_STRIKES_RANGE)
            end_idx = min(len(strikes), atm_idx + Config.ATM_STRIKES_RANGE + 1)
            selected_strikes = strikes[start_idx:end_idx]
            
            logger.info(f"✅ Selected {len(selected_strikes)} strikes (ATM ± {Config.ATM_STRIKES_RANGE})")
            logger.info(f"📊 Range: {min(selected_strikes):.0f} to {max(selected_strikes):.0f}")
            
            oi_list = []
            
            for strike in selected_strikes:
                try:
                    # Find strike in original data (as string with decimals)
                    strike_str = f"{strike:.6f}"
                    strike_data = oc_data.get(strike_str)
                    
                    if not strike_data:
                        # Try without decimals
                        strike_str = f"{int(strike)}.000000"
                        strike_data = oc_data.get(strike_str)
                    
                    if not strike_data:
                        continue
                    
                    ce_data = strike_data.get('ce', {})
                    pe_data = strike_data.get('pe', {})
                    
                    ce_oi = ce_data.get('oi', 0)
                    pe_oi = pe_data.get('oi', 0)
                    
                    oi_list.append(OIData(
                        strike=strike,
                        ce_oi=ce_oi,
                        pe_oi=pe_oi,
                        ce_volume=ce_data.get('volume', 0),
                        pe_volume=pe_data.get('volume', 0),
                        ce_iv=ce_data.get('implied_volatility', 0.0),
                        pe_iv=pe_data.get('implied_volatility', 0.0),
                        pcr_at_strike=pe_oi / ce_oi if ce_oi > 0 else 0
                    ))
                except Exception as e:
                    logger.error(f"❌ Strike {strike} parse error: {e}")
                    continue
            
            logger.info(f"✅ Fetched {len(oi_list)} strikes with OI data")
            
            return oi_list
            
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            logger.error(traceback.format_exc())
            return None


class AdvancedIndexBot:
    def __init__(self):
        logger.info("="*70)
        logger.info("🚀 NIFTY 50 BOT v10.4 - WEEKLY EXPIRY FIX")
        logger.info("="*70)
        
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.running = True
        
        self.total_scans = 0
        self.alerts_sent = 0
        
        logger.info("✅ Bot v10.4 initialized")
    
    def is_market_open(self) -> bool:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        return Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE
    
    async def scan_nifty(self):
        try:
            self.total_scans += 1
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 SCANNING NIFTY 50 - #{self.total_scans}")
            logger.info(f"{'='*70}")
            
            # ✅ Get WEEKLY expiry
            expiry = self.dhan.get_nearest_weekly_expiry()
            if not expiry:
                logger.warning("⚠️ No weekly expiry found")
                return
            
            # Get chart data
            mtf_data = self.dhan.get_multi_timeframe_data()
            if not mtf_data or '5m' not in mtf_data:
                logger.warning("⚠️ No chart data")
                return
            
            spot_price = mtf_data['5m']['close'].iloc[-1]
            logger.info(f"💰 Spot Price: {spot_price:.2f}")
            
            # Get option chain
            oi_data = self.dhan.get_option_chain(expiry, spot_price)
            if not oi_data or len(oi_data) < 10:
                logger.warning("⚠️ Insufficient option data")
                return
            
            # OI comparison
            oi_comparison = self.redis.get_oi_comparison("NIFTY 50", oi_data, spot_price)
            self.redis.store_option_chain("NIFTY 50", oi_data, spot_price)
            
            aggregate = oi_comparison.get('aggregate_analysis')
            if aggregate:
                logger.info(f"📊 OI: CE {aggregate.ce_oi_change_pct:+.2f}%, PE {aggregate.pe_oi_change_pct:+.2f}%")
                logger.info(f"📊 PCR: {aggregate.pcr:.2f} | Max Pain: {aggregate.max_pain:.0f}")
                logger.info(f"📊 Sentiment: {aggregate.overall_sentiment}")
                
                # Send alert if strong signal
                if abs(aggregate.ce_oi_change_pct) > 5 or abs(aggregate.pe_oi_change_pct) > 5:
                    await self.send_alert(spot_price, aggregate, expiry, oi_data)
                    self.alerts_sent += 1
            else:
                logger.info("📊 First scan - no OI comparison")
            
            logger.info(f"{'='*70}")
            logger.info(f"✅ Scan #{self.total_scans} complete | Alerts sent: {self.alerts_sent}")
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Scan error: {e}")
            logger.error(traceback.format_exc())
    
    async def send_alert(self, spot_price: float, aggregate: AggregateOIAnalysis, 
                        expiry: str, oi_data: List[OIData]):
        try:
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
            
            signal = "🟢 BULLISH" if aggregate.overall_sentiment == "BULLISH" else "🔴 BEARISH"
            
            # Find max OI strikes
            max_ce_strike = max(oi_data, key=lambda x: x.ce_oi)
            max_pe_strike = max(oi_data, key=lambda x: x.pe_oi)
            
            msg = f"""🔥 NIFTY 50 WEEKLY ALERT v10.4

{signal}
💰 Spot: {spot_price:.2f}
📅 Expiry: {expiry}

📊 OI ANALYSIS
━━━━━━━━━━━━━━━━━━
CE Change: {aggregate.ce_oi_change_pct:+.2f}%
PE Change: {aggregate.pe_oi_change_pct:+.2f}%
PCR: {aggregate.pcr:.2f}
Max Pain: {aggregate.max_pain:.0f}

🎯 KEY STRIKES
━━━━━━━━━━━━━━━━━━
Max CE OI: {max_ce_strike.strike:.0f} ({max_ce_strike.ce_oi:,})
Max PE OI: {max_pe_strike.strike:.0f} ({max_pe_strike.pe_oi:,})

⏰ {ist_time} IST"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg
            )
            
            logger.info("✅ Alert sent!")
            
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")
    
    async def send_startup_message(self):
        try:
            msg = f"""🚀 NIFTY 50 BOT v10.4 - WEEKLY EXPIRY FIX

✅ WEEKLY Expiry Detection
✅ Correct Strike Range
✅ ATM Detection Fixed
✅ Redis Caching
✅ Real-time OI Tracking

⏰ Scan Interval: 5 minutes
📊 Strike Range: ATM ± 10
🎯 Focus: Weekly Options

Status: 🟢 RUNNING"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg
            )
            logger.info("✅ Startup message sent!")
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")
    
    async def run(self):
        await self.send_startup_message()
        
        logger.info("\n" + "="*70)
        logger.info("🟢 BOT RUNNING")
        logger.info("="*70 + "\n")
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("⏸️ Market closed - waiting...")
                    await asyncio.sleep(60)
                    continue
                
                await self.scan_nifty()
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)


async def main():
    try:
        bot = AdvancedIndexBot()
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 NIFTY 50 BOT v10.4 - STARTING")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        logger.error(traceback.format_exc())
