"""
🤖 NIFTY 50 TRADING BOT v10.0 - ULTIMATE FIX
✅ Auto Security ID Detection from CSV
✅ Auto Nearest Expiry Selection
✅ ATM ± 10 strikes (21 total strikes)
✅ Fixed Option Chain Parsing
"""

import asyncio
import os
import json
import csv
import io
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
    """Bot Configuration - NIFTY 50 v10.0"""
    
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    DHAN_API_BASE = "https://api.dhan.co"
    DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
    DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"
    DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
    DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
    DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
    
    SCAN_INTERVAL = 300  # 5 minutes
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 86400
    
    CONFIDENCE_THRESHOLD = 70
    MIN_OI_DIVERGENCE_PCT = 3.0
    MIN_VOLUME_INCREASE_PCT = 30.0
    PCR_BULLISH_MIN = 1.1
    PCR_BEARISH_MAX = 0.9
    MIN_TOTAL_OI = 100000
    SKIP_OPENING_MINUTES = 10
    SKIP_CLOSING_MINUTES = 20
    
    LOOKBACK_DAYS = 15
    ATM_STRIKES_RANGE = 10  # ATM ± 10 = 21 strikes total
    MIN_CANDLES_REQUIRED = 50
    
    # ✅ Auto-detect from CSV
    INDEX_NAME = "NIFTY 50"
    INDEX_SECURITY_ID = None  # Will be loaded from CSV
    INDEX_SEGMENT = "IDX_I"
    FNO_SECURITY_ID = None  # Will be loaded from CSV
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
    ce_volume_change_pct: float
    pe_volume_change_pct: float
    pcr: float
    overall_sentiment: str
    max_pain: float
    max_pain_distance: float


@dataclass
class AdvancedAnalysis:
    opportunity: str
    confidence: int
    chart_score: int
    option_score: int
    alignment_score: int
    total_score: int
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
    divergence_warning: str
    scenario_bullish: str
    scenario_bearish: str
    risk_factors: List[str]
    monitoring_checklist: List[str]


class SecurityIDLoader:
    """CSV madhun Security IDs load karnar"""
    
    @staticmethod
    def load_nifty_ids() -> Dict:
        """NIFTY 50 cha INDEX ani FNO security IDs load karnar"""
        try:
            logger.info("🔍 Loading NIFTY 50 Security IDs from CSV...")
            
            response = requests.get(Config.DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"CSV download failed: {response.status_code}")
                return None
            
            csv_data = response.text.split('\n')
            reader = csv.DictReader(csv_data)
            
            index_id = None
            fno_id = None
            
            logger.info("Searching CSV for NIFTY 50...")
            
            for row in reader:
                try:
                    segment = row.get('SEM_SEGMENT', '').strip()
                    trading_symbol = row.get('SEM_TRADING_SYMBOL', '').strip()
                    exch_id = row.get('SEM_EXM_EXCH_ID', '').strip()
                    instrument_name = row.get('SEM_INSTRUMENT_NAME', '').strip()
                    custom_symbol = row.get('SEM_CUSTOM_SYMBOL', '').strip()
                    sec_id = row.get('SEM_SMST_SECURITY_ID', '').strip()
                    
                    # INDEX Security ID (for spot price)
                    # Try multiple variations: "NIFTY 50", "Nifty 50", "NIFTY50", etc.
                    if segment == 'I':
                        if 'NIFTY' in trading_symbol.upper() and '50' in trading_symbol:
                            if sec_id:
                                index_id = int(sec_id)
                                logger.info(f"✅ NIFTY 50 INDEX found: {trading_symbol} | ID: {index_id}")
                                logger.info(f"   Custom Symbol: {custom_symbol}")
                    
                    # F&O Security ID (for options)
                    # Look for: Segment D, NIFTY (not NIFTY BANK), NSE, OPTIDX
                    if segment == 'D' and exch_id == 'NSE':
                        if trading_symbol == 'NIFTY' and instrument_name == 'OPTIDX':
                            if sec_id:
                                fno_id = int(sec_id)
                                logger.info(f"✅ NIFTY F&O found: {trading_symbol} | ID: {fno_id}")
                                logger.info(f"   Instrument: {instrument_name}")
                    
                    # Agar dono mil gaye to break
                    if index_id and fno_id:
                        break
                        
                except Exception as e:
                    continue
            
            # If not found, try alternative approach - use hardcoded known IDs
            if not index_id or not fno_id:
                logger.warning("⚠️ CSV search failed, using fallback IDs...")
                
                # Known NIFTY 50 IDs from Dhan documentation
                if not index_id:
                    index_id = 13  # NIFTY 50 Index ID (IDX_I segment)
                    logger.info(f"📌 Using fallback INDEX ID: {index_id}")
                
                if not fno_id:
                    fno_id = 13  # NIFTY 50 F&O ID (NSE_FNO segment)
                    logger.info(f"📌 Using fallback F&O ID: {fno_id}")
            
            if index_id and fno_id:
                logger.info(f"✅ Final IDs - Index: {index_id}, F&O: {fno_id}")
                return {
                    'index_id': index_id,
                    'fno_id': fno_id
                }
            else:
                logger.error("❌ Could not find NIFTY 50 IDs!")
                return None
                
        except Exception as e:
            logger.error(f"Error loading security IDs: {e}")
            logger.error(traceback.format_exc())
            
            # Emergency fallback
            logger.warning("🚨 Emergency fallback - using known IDs")
            return {
                'index_id': 13,
                'fno_id': 13
            }


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
            logger.info("Redis connected!")
        except Exception as e:
            logger.error(f"Redis failed: {e}")
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
                        'pe_volume': oi.pe_volume,
                        'ce_iv': oi.ce_iv,
                        'pe_iv': oi.pe_iv
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
            previous_price = old_data.get('spot_price', current_price)
            
            total_ce_oi_old = sum(s['ce_oi'] for s in old_data['strikes'])
            total_pe_oi_old = sum(s['pe_oi'] for s in old_data['strikes'])
            total_ce_volume_old = sum(s['ce_volume'] for s in old_data['strikes'])
            total_pe_volume_old = sum(s['pe_volume'] for s in old_data['strikes'])
            
            total_ce_oi_new = sum(oi.ce_oi for oi in current_oi)
            total_pe_oi_new = sum(oi.pe_oi for oi in current_oi)
            total_ce_volume_new = sum(oi.ce_volume for oi in current_oi)
            total_pe_volume_new = sum(oi.pe_volume for oi in current_oi)
            
            ce_oi_change = total_ce_oi_new - total_ce_oi_old
            pe_oi_change = total_pe_oi_new - total_pe_oi_old
            
            ce_oi_change_pct = (ce_oi_change / total_ce_oi_old * 100) if total_ce_oi_old > 0 else 0
            pe_oi_change_pct = (pe_oi_change / total_pe_oi_old * 100) if total_pe_oi_old > 0 else 0
            
            ce_volume_change = total_ce_volume_new - total_ce_volume_old
            pe_volume_change = total_pe_volume_new - total_pe_volume_old
            
            ce_volume_change_pct = (ce_volume_change / total_ce_volume_old * 100) if total_ce_volume_old > 0 else 0
            pe_volume_change_pct = (pe_volume_change / total_pe_volume_old * 100) if total_pe_volume_old > 0 else 0
            
            pcr = total_pe_oi_new / total_ce_oi_new if total_ce_oi_new > 0 else 0
            
            sentiment = "NEUTRAL"
            if pe_oi_change_pct > 3 and pe_oi_change_pct > ce_oi_change_pct:
                sentiment = "BULLISH"
            elif ce_oi_change_pct > 3 and ce_oi_change_pct > pe_oi_change_pct:
                sentiment = "BEARISH"
            elif pcr > 1.2:
                sentiment = "BULLISH"
            elif pcr < 0.8:
                sentiment = "BEARISH"
            
            max_pain = self.calculate_max_pain(current_oi, current_price)
            max_pain_distance = ((current_price - max_pain) / current_price) * 100
            
            aggregate_analysis = AggregateOIAnalysis(
                total_ce_oi=total_ce_oi_new,
                total_pe_oi=total_pe_oi_new,
                total_ce_volume=total_ce_volume_new,
                total_pe_volume=total_pe_volume_new,
                total_ce_oi_change=ce_oi_change,
                total_pe_oi_change=pe_oi_change,
                ce_oi_change_pct=ce_oi_change_pct,
                pe_oi_change_pct=pe_oi_change_pct,
                ce_volume_change_pct=ce_volume_change_pct,
                pe_volume_change_pct=pe_volume_change_pct,
                pcr=pcr,
                overall_sentiment=sentiment,
                max_pain=max_pain,
                max_pain_distance=max_pain_distance
            )
            
            return {
                'change': 'UPDATED',
                'aggregate_analysis': aggregate_analysis,
                'price_change': current_price - previous_price,
                'old_spot': previous_price
            }
            
        except Exception as e:
            logger.error(f"Redis comparison error: {e}")
            return {'change': 'ERROR', 'aggregate_analysis': None}
    
    def calculate_max_pain(self, oi_data: List[OIData], spot_price: float) -> float:
        """Calculate Max Pain"""
        try:
            max_pain_strike = spot_price
            min_total_loss = float('inf')
            
            for test_strike_data in oi_data:
                test_strike = test_strike_data.strike
                total_loss = 0
                
                for oi in oi_data:
                    if spot_price > oi.strike:
                        total_loss += oi.ce_oi * (spot_price - oi.strike)
                    
                    if spot_price < oi.strike:
                        total_loss += oi.pe_oi * (oi.strike - spot_price)
                
                if total_loss < min_total_loss:
                    min_total_loss = total_loss
                    max_pain_strike = test_strike
            
            return max_pain_strike
        except Exception as e:
            logger.error(f"Max Pain error: {e}")
            return spot_price


class AdvancedChartAnalyzer:
    @staticmethod
    def identify_market_structure(df: pd.DataFrame) -> Dict:
        """Identify Market Structure"""
        try:
            if len(df) < 20:
                return {"structure": "INSUFFICIENT_DATA", "bias": "NEUTRAL"}
            
            recent = df.tail(50)
            highs = recent['high'].values
            lows = recent['low'].values
            
            swing_highs = []
            swing_lows = []
            
            for i in range(5, len(recent) - 5):
                if all(highs[i] >= highs[i-j] for j in range(1, 6)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, 6)):
                    swing_highs.append(highs[i])
                
                if all(lows[i] <= lows[i-j] for j in range(1, 6)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, 6)):
                    swing_lows.append(lows[i])
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                recent_highs = swing_highs[-2:]
                recent_lows = swing_lows[-2:]
                
                if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
                    return {"structure": "HH_HL", "bias": "BULLISH", "strength": "STRONG"}
                elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
                    return {"structure": "LH_LL", "bias": "BEARISH", "strength": "STRONG"}
            
            return {"structure": "SIDEWAYS", "bias": "NEUTRAL", "strength": "WEAK"}
        
        except Exception as e:
            logger.error(f"Market structure error: {e}")
            return {"structure": "ERROR", "bias": "NEUTRAL"}
    
    @staticmethod
    def find_order_blocks(df: pd.DataFrame) -> Dict:
        """Find Order Blocks"""
        try:
            if len(df) < 30:
                return {"bullish_ob": [], "bearish_ob": []}
            
            recent = df.tail(100)
            bullish_obs = []
            bearish_obs = []
            
            for i in range(10, len(recent) - 5):
                if recent['close'].iloc[i] < recent['open'].iloc[i]:
                    next_5_high = recent['high'].iloc[i+1:i+6].max()
                    if next_5_high > recent['high'].iloc[i] * 1.005:
                        bullish_obs.append({
                            'level': (recent['low'].iloc[i] + recent['high'].iloc[i]) / 2,
                            'strength': 'MODERATE'
                        })
                
                if recent['close'].iloc[i] > recent['open'].iloc[i]:
                    next_5_low = recent['low'].iloc[i+1:i+6].min()
                    if next_5_low < recent['low'].iloc[i] * 0.995:
                        bearish_obs.append({
                            'level': (recent['low'].iloc[i] + recent['high'].iloc[i]) / 2,
                            'strength': 'MODERATE'
                        })
            
            return {
                "bullish_ob": bullish_obs[-3:],
                "bearish_ob": bearish_obs[-3:]
            }
        
        except Exception as e:
            logger.error(f"Order blocks error: {e}")
            return {"bullish_ob": [], "bearish_ob": []}
    
    @staticmethod
    def calculate_multi_touch_sr(df: pd.DataFrame) -> Dict:
        """Calculate Support/Resistance"""
        try:
            if len(df) < 50:
                current = df['close'].iloc[-1]
                return {
                    'supports': [current * 0.98],
                    'resistances': [current * 1.02],
                    'support_tests': [1],
                    'resistance_tests': [1]
                }
            
            recent = df.tail(150)
            current = recent['close'].iloc[-1]
            
            highs = recent['high'].values
            lows = recent['low'].values
            
            resistance_levels = []
            support_levels = []
            
            window = 7
            for i in range(window, len(recent) - window):
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(highs[i])
                
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    support_levels.append(lows[i])
            
            def cluster_levels(levels, tolerance=0.005):
                if not levels:
                    return [], []
                
                levels = sorted(levels)
                clustered = []
                tests = []
                current_cluster = [levels[0]]
                
                for level in levels[1:]:
                    if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                        current_cluster.append(level)
                    else:
                        clustered.append(np.mean(current_cluster))
                        tests.append(len(current_cluster))
                        current_cluster = [level]
                
                clustered.append(np.mean(current_cluster))
                tests.append(len(current_cluster))
                return clustered, tests
            
            resistances, r_tests = cluster_levels(resistance_levels)
            supports, s_tests = cluster_levels(support_levels)
            
            resistances_filtered = [(r, t) for r, t in zip(resistances, r_tests) 
                                   if 0.001 <= (r - current)/current <= 0.05]
            supports_filtered = [(s, t) for s, t in zip(supports, s_tests) 
                                if 0.001 <= (current - s)/current <= 0.05]
            
            return {
                'supports': [s[0] for s in supports_filtered] if supports_filtered else [current * 0.98],
                'resistances': [r[0] for r in resistances_filtered] if resistances_filtered else [current * 1.02],
                'support_tests': [s[1] for s in supports_filtered] if supports_filtered else [1],
                'resistance_tests': [r[1] for r in resistances_filtered] if resistances_filtered else [1]
            }
        
        except Exception as e:
            logger.error(f"Multi-touch S/R error: {e}")
            current = df['close'].iloc[-1]
            return {
                'supports': [current * 0.98],
                'resistances': [current * 1.02],
                'support_tests': [1],
                'resistance_tests': [1]
            }


class DhanAPI:
    def __init__(self, redis_cache: RedisCache):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json'
        }
        self.redis = redis_cache
        logger.info(f"DhanAPI initialized - NIFTY 50 v10.0")
    
    def get_nearest_expiry(self) -> Optional[str]:
        """✅ AUTO: Get nearest expiry"""
        try:
            payload = {
                "UnderlyingScrip": Config.FNO_SECURITY_ID,
                "UnderlyingSeg": Config.FNO_SEGMENT
            }
            
            logger.info(f"Fetching expiry: {payload}")
            
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
                    nearest = expiries[0]
                    logger.info(f"✅ Auto-selected Expiry: {nearest}")
                    logger.info(f"Available expiries: {expiries}")
                    return nearest
            
            logger.error(f"Expiry fetch failed: {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"Expiry error: {e}")
            return None
    
    def get_multi_timeframe_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch MTF data"""
        try:
            logger.info(f"Fetching MTF data for NIFTY 50")
            
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
                logger.error(f"Intraday failed: {response.text}")
                return None
            
            data = response.json()
            
            if 'timestamp' not in data or len(data.get('open', [])) == 0:
                logger.error(f"No candle data")
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
            
            logger.info(f"Received {len(df_base)} candles")
            
            if len(df_base) < Config.MIN_CANDLES_REQUIRED:
                logger.warning(f"Only {len(df_base)} candles")
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
            
            result['1h'] = df_base.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.info(f"✅ MTF: 5m={len(result['5m'])}, 15m={len(result['15m'])}, 1h={len(result['1h'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"MTF data error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_option_chain(self, expiry: str, spot_price: float) -> Optional[List[OIData]]:
        """✅ ATM ± 10 strikes (21 total)"""
        try:
            payload = {
                "UnderlyingScrip": Config.FNO_SECURITY_ID,
                "UnderlyingSeg": Config.FNO_SEGMENT,
                "Expiry": expiry
            }
            
            logger.info(f"Fetching option chain: {payload}")
            
            response = requests.post(
                Config.DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                logger.error(f"Option chain failed: {response.text}")
                return None
            
            data = response.json()
            if not data.get('data'):
                logger.error("No option chain data")
                return None
            
            oc_data = data['data'].get('oc', {})
            
            if not oc_data:
                logger.error("No strikes")
                return None
            
            strikes = sorted([float(s) for s in oc_data.keys()])
            logger.info(f"Total strikes: {len(strikes)}")
            logger.info(f"Strike range: {min(strikes):.0f} to {max(strikes):.0f}")
            logger.info(f"Spot: {spot_price:.2f}")
            
            # ATM find karo
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            atm_idx = strikes.index(atm_strike)
            
            logger.info(f"✅ ATM: {atm_strike:.0f} (Index: {atm_idx})")
            logger.info(f"ATM distance: {abs(atm_strike - spot_price):.2f}")
            
            # ATM ± 10 = 21 strikes
            start_idx = max(0, atm_idx - Config.ATM_STRIKES_RANGE)
            end_idx = min(len(strikes), atm_idx + Config.ATM_STRIKES_RANGE + 1)
            selected_strikes = strikes[start_idx:end_idx]
            
            logger.info(f"✅ Selected {len(selected_strikes)} strikes (ATM ± {Config.ATM_STRIKES_RANGE})")
            logger.info(f"Range: {min(selected_strikes):.0f} to {max(selected_strikes):.0f}")
            
            oi_list = []
            
            for strike in selected_strikes:
                try:
                    strike_str = f"{strike:.6f}"
                    strike_data = oc_data.get(strike_str, {})
                    
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
                        ce_iv=ce_data.get('iv', 0.0),
                        pe_iv=pe_data.get('iv', 0.0),
                        pcr_at_strike=pe_oi / ce_oi if ce_oi > 0 else 0
                    ))
                except Exception as e:
                    logger.error(f"Strike {strike} parse error: {e}")
                    continue
            
            logger.info(f"✅ NIFTY 50: {len(oi_list)} strikes fetched")
            return oi_list
            
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            logger.error(traceback.format_exc())
            return None


class DeepSeekAdvancedAnalyzer:
    @staticmethod
    def create_advanced_analysis(spot_price: float, mtf_data: Dict,
                                 oi_data: List[OIData], oi_comparison: Dict,
                                 structure: Dict, order_blocks: Dict, sr_levels: Dict) -> Optional[AdvancedAnalysis]:
        try:
            logger.info(f"DeepSeek: Advanced analysis...")
            
            aggregate = oi_comparison.get('aggregate_analysis')
            
            if not aggregate:
                logger.warning("No aggregate OI data")
                return None
            
            structure_text = f"Structure: {structure.get('structure', 'N/A')} | Bias: {structure.get('bias', 'NEUTRAL')}"
            
            ob_text = "Order Blocks:\n"
            for ob in order_blocks.get('bullish_ob', [])[:2]:
                ob_text += f"  Bullish: {ob['level']:.0f}\n"
            for ob in order_blocks.get('bearish_ob', [])[:2]:
                ob_text += f"  Bearish: {ob['level']:.0f}\n"
            
            sr_text = "Support/Resistance:\n"
            for s in sr_levels['supports'][:3]:
                sr_text += f"  Support: {s:.0f}\n"
            for r in sr_levels['resistances'][:3]:
                sr_text += f"  Resistance: {r:.0f}\n"
            
            oi_text = f"""Option Chain:
CE OI: {aggregate.total_ce_oi:,} ({aggregate.ce_oi_change_pct:+.2f}%)
PE OI: {aggregate.total_pe_oi:,} ({aggregate.pe_oi_change_pct:+.2f}%)
PCR: {aggregate.pcr:.2f}
Max Pain: {aggregate.max_pain:.0f}
Sentiment: {aggregate.overall_sentiment}"""
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Expert F&O trader analyzing NIFTY 50.

Spot: {spot_price:.2f}

{structure_text}
{ob_text}
{sr_text}
{oi_text}

Analyze and reply JSON:
{{
  "opportunity": "PE_BUY or CE_BUY or WAIT",
  "confidence": 75,
  "chart_score": 40,
  "option_score": 45,
  "alignment_score": 20,
  "total_score": 105,
  "entry_price": {spot_price:.2f},
  "stop_loss": {spot_price * 0.995:.2f},
  "target_1": {spot_price * 1.01:.2f},
  "target_2": {spot_price * 1.02:.2f},
  "risk_reward": "1:2",
  "recommended_strike": {int(spot_price)},
  "pattern_signal": "Chart pattern",
  "oi_flow_signal": "OI flow",
  "market_structure": "HH_HL or LH_LL",
  "support_levels": [24500, 24400],
  "resistance_levels": [24700, 24800],
  "divergence_warning": "Any conflicts",
  "scenario_bullish": "If breaks X",
  "scenario_bearish": "If breaks Y",
  "risk_factors": ["Risk1", "Risk2"],
  "monitoring_checklist": ["Check1", "Check2"]
}}"""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "F&O expert. Reply ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"DeepSeek error: {response.status_code}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            analysis_dict = DeepSeekAdvancedAnalyzer.extract_json(content)
            
            if not analysis_dict:
                logger.error("Failed to extract JSON")
                return None
            
            analysis = AdvancedAnalysis(
                opportunity=analysis_dict.get('opportunity', 'WAIT'),
                confidence=analysis_dict.get('confidence', 50),
                chart_score=analysis_dict.get('chart_score', 0),
                option_score=analysis_dict.get('option_score', 0),
                alignment_score=analysis_dict.get('alignment_score', 0),
                total_score=analysis_dict.get('total_score', 0),
                entry_price=analysis_dict.get('entry_price', spot_price),
                stop_loss=analysis_dict.get('stop_loss', spot_price * 0.995),
                target_1=analysis_dict.get('target_1', spot_price * 1.01),
                target_2=analysis_dict.get('target_2', spot_price * 1.02),
                risk_reward=analysis_dict.get('risk_reward', '1:2'),
                recommended_strike=analysis_dict.get('recommended_strike', int(spot_price)),
                pattern_signal=analysis_dict.get('pattern_signal', 'N/A'),
                oi_flow_signal=analysis_dict.get('oi_flow_signal', 'N/A'),
                market_structure=analysis_dict.get('market_structure', 'SIDEWAYS'),
                support_levels=analysis_dict.get('support_levels', [spot_price * 0.98]),
                resistance_levels=analysis_dict.get('resistance_levels', [spot_price * 1.02]),
                divergence_warning=analysis_dict.get('divergence_warning', 'None'),
                scenario_bullish=analysis_dict.get('scenario_bullish', 'N/A'),
                scenario_bearish=analysis_dict.get('scenario_bearish', 'N/A'),
                risk_factors=analysis_dict.get('risk_factors', ['See analysis']),
                monitoring_checklist=analysis_dict.get('monitoring_checklist', ['Monitor'])
            )
            
            logger.info(f"DeepSeek: {analysis.opportunity} | {analysis.confidence}% | {analysis.total_score}/125")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def extract_json(content: str) -> Optional[Dict]:
        try:
            try:
                return json.loads(content)
            except:
                pass
            
            patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{[^{]*?"opportunity".*?\})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except:
                        continue
            
            start_idx = content.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                return json.loads(content[start_idx:i+1])
                            except:
                                break
            
            return None
            
        except Exception as e:
            logger.error(f"JSON error: {e}")
            return None


class ChartGenerator:
    @staticmethod
    def create_chart(mtf_data: Dict, analysis: AdvancedAnalysis) -> Optional[BytesIO]:
        try:
            logger.info(f"Generating chart")
            
            chart_df = mtf_data['15m'].tail(100).copy()
            
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick='inherit',
                volume='in',
                alpha=0.9
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='lightgray',
                facecolor='white',
                figcolor='white',
                y_on_right=False
            )
            
            hlines = dict(
                hlines=[analysis.entry_price, analysis.target_1, analysis.target_2, analysis.stop_loss],
                colors=['blue', 'green', 'darkgreen', 'red'],
                linestyle='--',
                linewidths=2
            )
            
            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style=s,
                title=f"NIFTY 50 v10.0 - {analysis.opportunity} (Score: {analysis.total_score}/125)",
                ylabel='Price',
                volume=True,
                hlines=hlines,
                returnfig=True,
                figsize=(16, 9),
                tight_layout=True
            )
            
            ax = axes[0]
            
            ax.text(len(chart_df), analysis.entry_price, f' Entry: {analysis.entry_price:.0f}', 
                   color='blue', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), analysis.target_1, f' T1: {analysis.target_1:.0f}', 
                   color='green', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), analysis.target_2, f' T2: {analysis.target_2:.0f}', 
                   color='darkgreen', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), analysis.stop_loss, f' SL: {analysis.stop_loss:.0f}', 
                   color='red', fontweight='bold', va='center', fontsize=10)
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            logger.info("Chart generated")
            return buf
            
        except Exception as e:
            logger.error(f"Chart error: {e}")
            return None


class AdvancedIndexBot:
    def __init__(self):
        logger.info("Initializing Bot v10.0 - NIFTY 50 ULTIMATE")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.chart_analyzer = AdvancedChartAnalyzer()
        self.deepseek = DeepSeekAdvancedAnalyzer()
        self.chart_gen = ChartGenerator()
        self.running = True
        
        self.total_scans = 0
        self.alerts_sent = 0
        
        logger.info("Bot v10.0 initialized")
    
    def is_market_open(self) -> bool:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        return Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE
    
    def escape_html(self, text: str) -> str:
        return html.escape(str(text))
    
    async def scan_nifty(self):
        try:
            self.total_scans += 1
            
            logger.info(f"\n{'='*70}")
            logger.info(f"SCANNING: NIFTY 50")
            logger.info(f"{'='*70}")
            
            expiry = self.dhan.get_nearest_expiry()
            if not expiry:
                logger.warning("No expiry found")
                return
            
            mtf_data = self.dhan.get_multi_timeframe_data()
            if not mtf_data or '5m' not in mtf_data:
                logger.warning("No chart data")
                return
            
            spot_price = mtf_data['5m']['close'].iloc[-1]
            logger.info(f"Spot: {spot_price:.2f}")
            
            structure = self.chart_analyzer.identify_market_structure(mtf_data['15m'])
            order_blocks = self.chart_analyzer.find_order_blocks(mtf_data['15m'])
            sr_levels = self.chart_analyzer.calculate_multi_touch_sr(mtf_data['15m'])
            
            logger.info(f"Structure: {structure.get('structure')} | Bias: {structure.get('bias')}")
            
            oi_data = self.dhan.get_option_chain(expiry, spot_price)
            if not oi_data or len(oi_data) < 10:
                logger.warning("Insufficient option data")
                return
            
            oi_comparison = self.redis.get_oi_comparison("NIFTY 50", oi_data, spot_price)
            self.redis.store_option_chain("NIFTY 50", oi_data, spot_price)
            
            aggregate = oi_comparison.get('aggregate_analysis')
            if aggregate:
                logger.info(f"OI: CE {aggregate.ce_oi_change_pct:+.2f}%, PE {aggregate.pe_oi_change_pct:+.2f}% | PCR {aggregate.pcr:.2f}")
            else:
                logger.info("First scan - no OI comparison")
            
            analysis = self.deepseek.create_advanced_analysis(
                spot_price, mtf_data, oi_data, oi_comparison,
                structure, order_blocks, sr_levels
            )
            
            if not analysis:
                logger.warning("No analysis")
                return
            
            if analysis.opportunity == "WAIT":
                logger.info("Analysis says WAIT")
                return
            
            if analysis.confidence < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"Confidence {analysis.confidence}% < {Config.CONFIDENCE_THRESHOLD}%")
                return
            
            ist = pytz.timezone('Asia/Kolkata')
            now_ist = datetime.now(ist)
            hour = now_ist.hour
            minute = now_ist.minute
            
            if hour == 9 and minute < 15 + Config.SKIP_OPENING_MINUTES:
                logger.info("Opening period - skip")
                return
            
            if hour == 15 or (hour == 14 and minute >= (60 - Config.SKIP_CLOSING_MINUTES)):
                logger.info("Closing period - skip")
                return
            
            chart_image = self.chart_gen.create_chart(mtf_data, analysis)
            
            await self.send_alert(spot_price, analysis, aggregate, expiry, chart_image)
            
            self.alerts_sent += 1
            logger.info(f"✅ ALERT SENT!")
            logger.info(f"Stats: Scans={self.total_scans}, Alerts={self.alerts_sent}")
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            logger.error(traceback.format_exc())
    
    async def send_alert(self, spot_price: float, analysis: AdvancedAnalysis,
                        aggregate: Optional[AggregateOIAnalysis], expiry: str, chart_image: Optional[BytesIO]):
        try:
            signal_map = {
                "PE_BUY": ("🔴", "PE BUY"),
                "CE_BUY": ("🟢", "CE BUY"),
                "WAIT": ("⚪", "WAIT")
            }
            
            signal_emoji, signal_text = signal_map.get(analysis.opportunity, ("⚪", "WAIT"))
            
            def safe(val):
                return self.escape_html(val)
            
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
            
            caption = f"🔥 NIFTY 50 v10.0\n\n{signal_emoji} {signal_text} | {analysis.confidence}%\nScore: {analysis.total_score}/125\n\n💰 Entry: {analysis.entry_price:.0f} | SL: {analysis.stop_loss:.0f}\n🎯 T1: {analysis.target_1:.0f} | T2: {analysis.target_2:.0f}\nRR: {analysis.risk_reward} | Strike: {analysis.recommended_strike}\n\n⏰ {ist_time} IST"
            
            if chart_image:
                try:
                    await self.bot.send_photo(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        photo=chart_image,
                        caption=caption
                    )
                except Exception as e:
                    logger.error(f"Chart send failed: {e}")
                    await self.bot.send_message(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        text=caption
                    )
            
            agg_text = ""
            if aggregate:
                agg_text = f"PCR: {aggregate.pcr:.2f} | Max Pain: {aggregate.max_pain:.0f}\nOI: CE {aggregate.ce_oi_change_pct:+.1f}% PE {aggregate.pe_oi_change_pct:+.1f}%"
            
            supports_text = ", ".join([f"{s:.0f}" for s in analysis.support_levels[:3]])
            resistances_text = ", ".join([f"{r:.0f}" for r in analysis.resistance_levels[:3]])
            
            detailed = f"""🔥 NIFTY 50 v10.0 ANALYSIS

{'='*40}
SCORING
{'='*40}
Chart: {analysis.chart_score}/50
Options: {analysis.option_score}/50
Alignment: {analysis.alignment_score}/25
TOTAL: {analysis.total_score}/125

{'='*40}
STRUCTURE
{'='*40}
{safe(analysis.market_structure)}

{'='*40}
LEVELS
{'='*40}
Support: {supports_text}
Resistance: {resistances_text}

{'='*40}
OPTIONS
{'='*40}
{agg_text}

{'='*40}
SIGNALS
{'='*40}
📊 {safe(analysis.pattern_signal[:150])}
⛓️ {safe(analysis.oi_flow_signal[:150])}

{'='*40}
SCENARIOS
{'='*40}
🟢 {safe(analysis.scenario_bullish[:120])}
🔴 {safe(analysis.scenario_bearish[:120])}

{'='*40}
RISKS
{'='*40}"""
            
            for risk in analysis.risk_factors[:3]:
                detailed += f"\n⚠️ {safe(risk[:80])}"
            
            detailed += f"\n\n{'='*40}\nMONITOR\n{'='*40}"
            
            for check in analysis.monitoring_checklist[:3]:
                detailed += f"\n✓ {safe(check[:80])}"
            
            if analysis.divergence_warning and analysis.divergence_warning != "None":
                detailed += f"\n\n⚠️ {safe(analysis.divergence_warning[:150])}"
            
            detailed += f"\n\n🤖 v10.0 | Expiry: {expiry}"
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=detailed,
                parse_mode='HTML'
            )
            
            logger.info("Alert sent!")
            return True
            
        except Exception as e:
            logger.error(f"Alert error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def send_startup_message(self):
        try:
            redis_status = "✅" if self.redis.redis_client else "❌"
            
            msg = f"""🔥 NIFTY 50 BOT v10.0 - ULTIMATE FIX 🔥

{'='*40}
✅ AUTO SECURITY ID FROM CSV
{'='*40}

📊 Index ID: {Config.INDEX_SECURITY_ID}
🎯 F&O ID: {Config.FNO_SECURITY_ID}
⚡ Auto Expiry: NEAREST
🎲 Strikes: ATM ± 10 (21 total)

{'='*40}
v10.0 FEATURES
{'='*40}
✅ Auto-detect Security IDs from CSV
✅ Auto-select nearest expiry
✅ ATM ± 10 strikes (21 total)
✅ Fixed option chain parsing
✅ Correct NIFTY 50 data

{'='*40}
SYSTEM
{'='*40}
⏰ Scan: Every 5 minutes
🔴 Redis: {redis_status}
🤖 AI: DeepSeek V3
📈 Market Structure Analysis
🎯 Order Blocks & S/R
💎 Max Pain & OI Flow

{'='*40}
FILTERS
{'='*40}
Confidence: ≥70%
PCR: >1.1 or <0.9
Time: Skip first 10m, last 20m

Status: 🟢 RUNNING"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg
            )
            logger.info("Startup sent!")
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    async def run(self):
        logger.info("="*70)
        logger.info("NIFTY 50 BOT v10.0 - ULTIMATE FIX")
        logger.info("="*70)
        
        # ✅ Load Security IDs from CSV
        logger.info("Loading Security IDs from CSV...")
        ids = SecurityIDLoader.load_nifty_ids()
        
        if not ids:
            logger.error("❌ Failed to load Security IDs!")
            return
        
        Config.INDEX_SECURITY_ID = ids['index_id']
        Config.FNO_SECURITY_ID = ids['fno_id']
        
        logger.info(f"✅ Security IDs loaded successfully!")
        logger.info(f"   Index ID: {Config.INDEX_SECURITY_ID}")
        logger.info(f"   F&O ID: {Config.FNO_SECURITY_ID}")
        
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            val = getattr(Config, cred)
            if not val:
                missing.append(cred)
            else:
                if 'TOKEN' in cred or 'KEY' in cred:
                    logger.info(f"{cred}: {val[:10]}...{val[-10:]}")
                else:
                    logger.info(f"{cred}: {val}")
        
        if missing:
            logger.error(f"❌ MISSING: {', '.join(missing)}")
            return
        
        logger.info("✅ All credentials present")
        
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("Bot RUNNING - v10.0 ULTIMATE")
        logger.info("="*70)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("Market closed")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*70}")
                logger.info(f"SCAN CYCLE - {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"{'='*70}")
                
                await self.scan_nifty()
                
                logger.info(f"\n{'='*70}")
                logger.info(f"CYCLE COMPLETE")
                logger.info(f"Stats: Scans={self.total_scans}, Alerts={self.alerts_sent}")
                logger.info(f"{'='*70}\n")
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Stopped")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)


async def main():
    try:
        bot = AdvancedIndexBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("NIFTY 50 BOT v10.0 - ULTIMATE FIX STARTING")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown")
    except Exception as e:
        logger.error(f"\nCritical: {e}")
        logger.error(traceback.format_exc())
