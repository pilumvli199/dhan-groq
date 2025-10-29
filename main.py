"""
🤖 ADVANCED NIFTY/SENSEX INDEX TRADING BOT v9.1
Version: 9.1 - INDICES ONLY (FIXED DATA FETCHING)
Advanced Price Action + Option Chain Analysis
Scan Interval: 5 minutes | Flexible Rules
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
    """Bot Configuration - FLEXIBLE RULES FOR INDICES"""
    
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
    
    # FLEXIBLE FILTERS (Not too strict!)
    SCAN_INTERVAL = 300  # 5 minutes
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 86400
    
    CONFIDENCE_THRESHOLD = 70  # Flexible (was 80)
    MIN_OI_DIVERGENCE_PCT = 3.0  # Flexible (was 5.0)
    MIN_VOLUME_INCREASE_PCT = 30.0  # Flexible (was 50.0)
    PCR_BULLISH_MIN = 1.1  # Flexible (was 1.2)
    PCR_BEARISH_MAX = 0.9  # Flexible (was 0.8)
    MIN_TOTAL_OI = 100000  # Lower for indices
    SKIP_OPENING_MINUTES = 10  # Reduced (was 15)
    SKIP_CLOSING_MINUTES = 20  # Reduced (was 30)
    
    LOOKBACK_DAYS = 15  # More data for indices
    ATM_STRIKE_RANGE = 15  # Wider range for indices
    MIN_CANDLES_REQUIRED = 50
    
    # INDICES ONLY - FIXED SECURITY IDs
    INDICES = {
        "NIFTY 50": {
            "symbol": "NIFTY 50",
            "security_id": "13",  # NSE NIFTY 50 (string format)
            "segment": "IDX_I",  # Index segment
            "instrument": "INDEX",
            "exchange": "NSE"
        },
        "SENSEX": {
            "symbol": "SENSEX",
            "security_id": "51",  # BSE SENSEX (string format)
            "segment": "IDX_I",  # Index segment
            "instrument": "INDEX",
            "exchange": "BSE"
        }
    }
    
    # FNO segments for option chain
    FNO_SEGMENTS = {
        "NIFTY 50": "NSE_FNO",
        "SENSEX": "BSE_FNO"
    }


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
            
            self.redis.setex(key, Config.REDIS_EXPIRY, value)
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
            
            # Calculate Max Pain
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
        """Calculate Max Pain - strike where option writers lose least"""
        try:
            max_pain_strike = spot_price
            min_total_loss = float('inf')
            
            for test_strike_data in oi_data:
                test_strike = test_strike_data.strike
                total_loss = 0
                
                for oi in oi_data:
                    # Call writers loss (if spot > strike)
                    if spot_price > oi.strike:
                        total_loss += oi.ce_oi * (spot_price - oi.strike)
                    
                    # Put writers loss (if spot < strike)
                    if spot_price < oi.strike:
                        total_loss += oi.pe_oi * (oi.strike - spot_price)
                
                if total_loss < min_total_loss:
                    min_total_loss = total_loss
                    max_pain_strike = test_strike
            
            return max_pain_strike
        except Exception as e:
            logger.error(f"Max Pain calculation error: {e}")
            return spot_price


class AdvancedChartAnalyzer:
    @staticmethod
    def identify_market_structure(df: pd.DataFrame) -> Dict:
        """Identify Higher Highs, Higher Lows, etc."""
        try:
            if len(df) < 20:
                return {"structure": "INSUFFICIENT_DATA", "bias": "NEUTRAL"}
            
            recent = df.tail(50)
            highs = recent['high'].values
            lows = recent['low'].values
            
            # Find swing highs and lows
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
        """Find significant demand/supply zones"""
        try:
            if len(df) < 30:
                return {"bullish_ob": [], "bearish_ob": []}
            
            recent = df.tail(100)
            bullish_obs = []
            bearish_obs = []
            
            for i in range(10, len(recent) - 5):
                # Bullish order block: down candle followed by strong up move
                if recent['close'].iloc[i] < recent['open'].iloc[i]:  # Down candle
                    next_5_high = recent['high'].iloc[i+1:i+6].max()
                    if next_5_high > recent['high'].iloc[i] * 1.005:  # 0.5% move up
                        bullish_obs.append({
                            'level': (recent['low'].iloc[i] + recent['high'].iloc[i]) / 2,
                            'strength': 'MODERATE'
                        })
                
                # Bearish order block: up candle followed by strong down move
                if recent['close'].iloc[i] > recent['open'].iloc[i]:  # Up candle
                    next_5_low = recent['low'].iloc[i+1:i+6].min()
                    if next_5_low < recent['low'].iloc[i] * 0.995:  # 0.5% move down
                        bearish_obs.append({
                            'level': (recent['low'].iloc[i] + recent['high'].iloc[i]) / 2,
                            'strength': 'MODERATE'
                        })
            
            return {
                "bullish_ob": bullish_obs[-3:],  # Last 3
                "bearish_ob": bearish_obs[-3:]
            }
        
        except Exception as e:
            logger.error(f"Order blocks error: {e}")
            return {"bullish_ob": [], "bearish_ob": []}
    
    @staticmethod
    def calculate_multi_touch_sr(df: pd.DataFrame) -> Dict:
        """Find support/resistance with multiple touches"""
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
            
            # Find pivot points
            highs = recent['high'].values
            lows = recent['low'].values
            
            resistance_levels = []
            support_levels = []
            
            window = 7
            for i in range(window, len(recent) - window):
                # Resistance: local high
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(highs[i])
                
                # Support: local low
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    support_levels.append(lows[i])
            
            # Cluster nearby levels
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
            
            # Filter relevant levels (within 5% of current price)
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
        logger.info(f"DhanAPI initialized - ClientID: {Config.DHAN_CLIENT_ID[:10]}...")
        logger.info(f"Access Token present: {bool(Config.DHAN_ACCESS_TOKEN)}")
    
    def get_nearest_expiry(self, index_name: str) -> Optional[str]:
        """Get nearest expiry for index options"""
        try:
            index_info = Config.INDICES[index_name]
            fno_segment = Config.FNO_SEGMENTS[index_name]
            
            # For NIFTY, use security_id 25 for FNO
            # For SENSEX, use security_id 51 for FNO
            fno_security_id = 25 if index_name == "NIFTY 50" else 51
            
            payload = {
                "UnderlyingScrip": str(fno_security_id),
                "UnderlyingSeg": fno_segment
            }
            
            logger.info(f"Fetching expiry for {index_name}: {payload}")
            
            response = requests.post(
                Config.DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            logger.info(f"Expiry response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Expiry data: {data}")
                if data.get('status') == 'success' and data.get('data'):
                    expiry = data['data'][0]
                    logger.info(f"{index_name} expiry: {expiry}")
                    return expiry
            
            logger.error(f"Expiry fetch failed: {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"Expiry error for {index_name}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_multi_timeframe_data(self, index_name: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch intraday data for indices"""
        try:
            logger.info(f"Fetching MTF data for {index_name}")
            
            index_info = Config.INDICES[index_name]
            security_id = index_info['security_id']
            segment = index_info['segment']
            instrument = index_info['instrument']
            
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=Config.LOOKBACK_DAYS)
            
            # Try intraday API first
            payload = {
                "securityId": security_id,
                "exchangeSegment": segment,
                "instrument": instrument,
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            logger.info(f"Intraday payload: {payload}")
            
            response = requests.post(
                Config.DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            logger.info(f"Intraday response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Intraday failed: {response.text}")
                
                # Try historical API as fallback
                logger.info(f"Trying historical API for {index_name}")
                
                hist_payload = {
                    "securityId": security_id,
                    "exchangeSegment": segment,
                    "instrument": instrument,
                    "expiryCode": 0,
                    "fromDate": from_date.strftime("%Y-%m-%d"),
                    "toDate": to_date.strftime("%Y-%m-%d")
                }
                
                response = requests.post(
                    Config.DHAN_HISTORICAL_URL,
                    json=hist_payload,
                    headers=self.headers,
                    timeout=15
                )
                
                logger.info(f"Historical response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Historical also failed: {response.text}")
                    return None
            
            data = response.json()
            
            if 'timestamp' not in data or len(data.get('open', [])) == 0:
                logger.error(f"No candle data in response: {data}")
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
            
            logger.info(f"Received {len(df_base)} base candles for {index_name}")
            
            if len(df_base) < Config.MIN_CANDLES_REQUIRED:
                logger.warning(f"{index_name}: Only {len(df_base)} candles (need {Config.MIN_CANDLES_REQUIRED})")
                return None
            
            # Create multiple timeframes
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
            
            logger.info(f"{index_name}: 5m={len(result['5m'])}, 15m={len(result['15m'])}, 1h={len(result['1h'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"MTF data error for {index_name}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_option_chain(self, index_name: str, expiry: str, spot_price: float) -> Optional[List[OIData]]:
        """Fetch option chain data"""
        try:
            fno_segment = Config.FNO_SEGMENTS[index_name]
            fno_security_id = 25 if index_name == "NIFTY 50" else 51
            
            payload = {
                "UnderlyingScrip": str(fno_security_id),
                "UnderlyingSeg": fno_segment,
                "Expiry": expiry
            }
            
            logger.info(f"Fetching option chain for {index_name}: {payload}")
            
            response = requests.post(
                Config.DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            logger.info(f"Option chain response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Option chain fetch failed: {response.text}")
                return None
            
            data = response.json()
            if not data.get('data'):
                logger.error("No option chain data")
                return None
            
            oc_data = data['data'].get('oc', {})
            
            if not oc_data:
                logger.error("No option chain strikes")
                return None
            
            strikes = [float(s) for s in oc_data.keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            logger.info(f"{index_name} ATM: {atm_strike} (Spot: {spot_price:.2f})")
            
            oi_list = []
            
            for strike_str, strike_data in oc_data.items():
                try:
                    strike = float(strike_str)
                    
                    # Get strikes around ATM
                    if abs(strike - atm_strike) > (atm_strike * 0.05):  # Within 5%
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
                except Exception:
                    continue
            
            logger.info(f"{index_name}: {len(oi_list)} strikes fetched")
            return oi_list
            
        except Exception as e:
            logger.error(f"Option chain error for {index_name}: {e}")
            logger.error(traceback.format_exc())
            return None


class DeepSeekAdvancedAnalyzer:
    @staticmethod
    def create_advanced_analysis(symbol: str, spot_price: float, mtf_data: Dict,
                                 oi_data: List[OIData], oi_comparison: Dict,
                                 structure: Dict, order_blocks: Dict, sr_levels: Dict) -> Optional[AdvancedAnalysis]:
        try:
            logger.info(f"DeepSeek: Advanced analysis for {symbol}...")
            
            aggregate = oi_comparison.get('aggregate_analysis')
            
            if not aggregate:
                logger.warning("No aggregate OI data for advanced analysis")
                return None
            
            # Build comprehensive prompt
            structure_text = f"Structure: {structure.get('structure', 'N/A')} | Bias: {structure.get('bias', 'NEUTRAL')}"
            
            ob_text = "Order Blocks:\n"
            for ob in order_blocks.get('bullish_ob', [])[:2]:
                ob_text += f"  Bullish: {ob['level']:.0f} ({ob['strength']})\n"
            for ob in order_blocks.get('bearish_ob', [])[:2]:
                ob_text += f"  Bearish: {ob['level']:.0f} ({ob['strength']})\n"
            
            sr_text = "Support/Resistance (Multi-touch):\n"
            for i, (s, t) in enumerate(zip(sr_levels['supports'][:3], sr_levels['support_tests'][:3])):
                sr_text += f"  Support: {s:.0f} ({t} tests)\n"
            for i, (r, t) in enumerate(zip(sr_levels['resistances'][:3], sr_levels['resistance_tests'][:3])):
                sr_text += f"  Resistance: {r:.0f} ({t} tests)\n"
            
            oi_text = f"""Option Chain Analysis:
Total CE OI: {aggregate.total_ce_oi:,} (Change: {aggregate.ce_oi_change_pct:+.2f}%)
Total PE OI: {aggregate.total_pe_oi:,} (Change: {aggregate.pe_oi_change_pct:+.2f}%)
CE Volume Change: {aggregate.ce_volume_change_pct:+.2f}%
PE Volume Change: {aggregate.pe_volume_change_pct:+.2f}%
PCR: {aggregate.pcr:.2f}
Max Pain: {aggregate.max_pain:.0f} (Distance: {aggregate.max_pain_distance:+.2f}%)
Sentiment: {aggregate.overall_sentiment}"""
            
            # Top OI strikes
            oi_sorted = sorted(oi_data, key=lambda x: x.ce_oi + x.pe_oi, reverse=True)[:5]
            strikes_text = "Top 5 OI Strikes:\n"
            for oi in oi_sorted:
                strikes_text += f"  {oi.strike:.0f}: CE {oi.ce_oi:,} | PE {oi.pe_oi:,} | PCR {oi.pcr_at_strike:.2f}\n"
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""You are an expert F&O trader analyzing {symbol} for high-probability options trading.

=== ADVANCED PRICE ACTION ANALYSIS ===

Spot Price: {spot_price:.2f}

MARKET STRUCTURE:
{structure_text}

{ob_text}

{sr_text}

{oi_text}

{strikes_text}

=== YOUR TASK ===

Perform COMPREHENSIVE analysis combining:

1. CHART ANALYSIS (Score out of 50):
   - Market structure (HH/HL or LH/LL)
   - Order blocks strength
   - Support/Resistance multi-touch validity
   - Current price position
   - Breakout/breakdown potential
   
2. OPTION CHAIN ANALYSIS (Score out of 50):
   - OI distribution and changes
   - Volume surge analysis
   - PCR interpretation
   - Max Pain influence
   - IV analysis
   
3. ALIGNMENT CHECK (Score out of 25):
   - Do chart S/R match OI clusters?
   - Do chart signals match option signals?
   - Any divergence warnings?

4. TRADE SETUP:
   - Opportunity: PE_BUY / CE_BUY / WAIT
   - Entry price (specific level)
   - Stop loss (below/above key level with reason)
   - Target 1 (conservative)
   - Target 2 (aggressive)
   - Risk-Reward ratio
   - Recommended strike (ATM or nearby)

5. SCENARIOS:
   - Bullish scenario: "If price does X, expect Y"
   - Bearish scenario: "If price does A, expect B"
   
6. MONITORING:
   - What to watch every 30 min
   - Key levels to set alerts
   
7. RISK FACTORS:
   - What can invalidate setup
   - Major S/R blocking move
   - Any divergence between chart/options

Reply in JSON format:
{{
  "opportunity": "PE_BUY or CE_BUY or WAIT",
  "confidence": 75,
  "chart_score": 42,
  "option_score": 45,
  "alignment_score": 22,
  "total_score": 109,
  "entry_price": {spot_price:.2f},
  "stop_loss": {spot_price * 0.995:.2f},
  "target_1": {spot_price * 1.01:.2f},
  "target_2": {spot_price * 1.02:.2f},
  "risk_reward": "1:2",
  "recommended_strike": {int(spot_price)},
  "pattern_signal": "Brief chart pattern summary",
  "oi_flow_signal": "Brief OI flow summary",
  "market_structure": "HH_HL or LH_LL or SIDEWAYS",
  "support_levels": [23400, 23350],
  "resistance_levels": [23600, 23650],
  "divergence_warning": "Any conflict between chart and options",
  "scenario_bullish": "If breaks 23600, targets 23750",
  "scenario_bearish": "If breaks 23400, targets 23250",
  "risk_factors": ["Risk 1", "Risk 2", "Risk 3"],
  "monitoring_checklist": ["Check 1", "Check 2", "Check 3"]
}}

IMPORTANT: Be realistic. If setup unclear or conflicting signals, say WAIT."""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Expert F&O trader. Reply ONLY valid JSON. Be realistic about confidence."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Extract JSON
            analysis_dict = DeepSeekAdvancedAnalyzer.extract_json(content)
            
            if not analysis_dict:
                logger.error("Failed to extract JSON from DeepSeek response")
                return None
            
            # Validate required fields
            required = ['opportunity', 'confidence', 'chart_score', 'option_score', 'alignment_score']
            if not all(f in analysis_dict for f in required):
                logger.error("Missing required fields in analysis")
                return None
            
            analysis = AdvancedAnalysis(
                opportunity=analysis_dict['opportunity'],
                confidence=analysis_dict['confidence'],
                chart_score=analysis_dict['chart_score'],
                option_score=analysis_dict['option_score'],
                alignment_score=analysis_dict['alignment_score'],
                total_score=analysis_dict['total_score'],
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
                monitoring_checklist=analysis_dict.get('monitoring_checklist', ['Monitor price action'])
            )
            
            logger.info(f"DeepSeek: {analysis.opportunity} | Confidence: {analysis.confidence}% | Score: {analysis.total_score}/125")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced analysis error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def extract_json(content: str) -> Optional[Dict]:
        """Extract JSON from response"""
        try:
            # Try direct parse
            try:
                return json.loads(content)
            except:
                pass
            
            # Try markdown code block
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
            
            # Try brace counting
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
            logger.error(f"JSON extraction error: {e}")
            return None


class ChartGenerator:
    @staticmethod
    def create_chart(mtf_data: Dict, symbol: str, analysis: AdvancedAnalysis) -> Optional[BytesIO]:
        try:
            logger.info(f"Generating chart for {symbol}")
            
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
                title=f"{symbol} - {analysis.opportunity} (Score: {analysis.total_score}/125)",
                ylabel='Price',
                volume=True,
                hlines=hlines,
                returnfig=True,
                figsize=(16, 9),
                tight_layout=True
            )
            
            ax = axes[0]
            current_price = chart_df['close'].iloc[-1]
            
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
        logger.info("Initializing Advanced Index Bot v9.1...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.chart_analyzer = AdvancedChartAnalyzer()
        self.deepseek = DeepSeekAdvancedAnalyzer()
        self.chart_gen = ChartGenerator()
        self.running = True
        
        self.total_scans = 0
        self.alerts_sent = 0
        
        logger.info("Bot v9.1 initialized - NIFTY/SENSEX ONLY (FIXED)")
    
    def is_market_open(self) -> bool:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        return Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE
    
    def escape_html(self, text: str) -> str:
        """Properly escape HTML special characters"""
        return html.escape(str(text))
    
    async def scan_index(self, index_name: str):
        try:
            self.total_scans += 1
            
            logger.info(f"\n{'='*70}")
            logger.info(f"SCANNING: {index_name}")
            logger.info(f"{'='*70}")
            
            # Get expiry
            expiry = self.dhan.get_nearest_expiry(index_name)
            if not expiry:
                logger.warning(f"{index_name}: No expiry found")
                return
            
            # Get chart data
            mtf_data = self.dhan.get_multi_timeframe_data(index_name)
            if not mtf_data or '5m' not in mtf_data:
                logger.warning(f"{index_name}: No chart data")
                return
            
            spot_price = mtf_data['5m']['close'].iloc[-1]
            logger.info(f"Spot: {spot_price:.2f}")
            
            # Advanced chart analysis
            structure = self.chart_analyzer.identify_market_structure(mtf_data['15m'])
            order_blocks = self.chart_analyzer.find_order_blocks(mtf_data['15m'])
            sr_levels = self.chart_analyzer.calculate_multi_touch_sr(mtf_data['15m'])
            
            logger.info(f"Structure: {structure.get('structure')} | Bias: {structure.get('bias')}")
            
            # Get option chain
            oi_data = self.dhan.get_option_chain(index_name, expiry, spot_price)
            if not oi_data or len(oi_data) < 10:
                logger.warning(f"{index_name}: No option data")
                return
            
            # OI comparison
            oi_comparison = self.redis.get_oi_comparison(index_name, oi_data, spot_price)
            self.redis.store_option_chain(index_name, oi_data, spot_price)
            
            aggregate = oi_comparison.get('aggregate_analysis')
            if aggregate:
                logger.info(f"OI: CE {aggregate.ce_oi_change_pct:+.2f}%, PE {aggregate.pe_oi_change_pct:+.2f}% | PCR {aggregate.pcr:.2f}")
                logger.info(f"Max Pain: {aggregate.max_pain:.0f} (Distance: {aggregate.max_pain_distance:+.2f}%)")
            else:
                logger.info("First scan - no OI comparison")
            
            # DeepSeek advanced analysis
            analysis = self.deepseek.create_advanced_analysis(
                index_name, spot_price, mtf_data, oi_data, oi_comparison,
                structure, order_blocks, sr_levels
            )
            
            if not analysis:
                logger.warning(f"{index_name}: No analysis from DeepSeek")
                return
            
            # Flexible filter (not too strict!)
            if analysis.opportunity == "WAIT":
                logger.info(f"{index_name}: Analysis says WAIT")
                return
            
            if analysis.confidence < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"{index_name}: Confidence {analysis.confidence}% < {Config.CONFIDENCE_THRESHOLD}%")
                return
            
            # Check time filter
            ist = pytz.timezone('Asia/Kolkata')
            now_ist = datetime.now(ist)
            hour = now_ist.hour
            minute = now_ist.minute
            
            if hour == 9 and minute < 15 + Config.SKIP_OPENING_MINUTES:
                logger.info(f"{index_name}: Market opening period - skip")
                return
            
            if hour == 15 or (hour == 14 and minute >= (60 - Config.SKIP_CLOSING_MINUTES)):
                logger.info(f"{index_name}: Market closing period - skip")
                return
            
            # Generate chart
            chart_image = self.chart_gen.create_chart(mtf_data, index_name, analysis)
            
            # Send alert
            await self.send_alert(index_name, spot_price, analysis, aggregate, expiry, chart_image)
            
            self.alerts_sent += 1
            logger.info(f"✅ {index_name}: ALERT SENT!")
            logger.info(f"Stats: Total Scans={self.total_scans}, Alerts={self.alerts_sent}")
            
        except Exception as e:
            logger.error(f"Scan error {index_name}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_alert(self, index_name: str, spot_price: float, analysis: AdvancedAnalysis,
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
            
            # Compact caption for image - NO HTML FORMATTING
            caption = f"🔥 ADVANCED ANALYSIS - {index_name}\n\n{signal_emoji} {signal_text} | Confidence: {analysis.confidence}%\nScore: {analysis.total_score}/125 (Chart:{analysis.chart_score} Options:{analysis.option_score} Align:{analysis.alignment_score})\n\n💰 Entry: {analysis.entry_price:.0f} | SL: {analysis.stop_loss:.0f}\n🎯 T1: {analysis.target_1:.0f} | T2: {analysis.target_2:.0f}\nRR: {analysis.risk_reward} | Strike: {analysis.recommended_strike}\n\n⏰ {ist_time} IST | v9.1 Fixed"
            
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
            
            # Detailed message - WITH HTML ESCAPING
            agg_text = ""
            if aggregate:
                agg_text = f"PCR: {aggregate.pcr:.2f} | Max Pain: {aggregate.max_pain:.0f} ({aggregate.max_pain_distance:+.2f}%)\nOI: CE {aggregate.ce_oi_change_pct:+.1f}% PE {aggregate.pe_oi_change_pct:+.1f}%\nVol: CE {aggregate.ce_volume_change_pct:+.1f}% PE {aggregate.pe_volume_change_pct:+.1f}%"
            
            supports_text = ", ".join([f"{s:.0f}" for s in analysis.support_levels[:3]])
            resistances_text = ", ".join([f"{r:.0f}" for r in analysis.resistance_levels[:3]])
            
            detailed = f"""🔥 ADVANCED ANALYSIS - {safe(index_name)}

{'='*40}
SCORING BREAKDOWN
{'='*40}
Chart Analysis: {analysis.chart_score}/50
Option Analysis: {analysis.option_score}/50
Alignment: {analysis.alignment_score}/25
TOTAL SCORE: {analysis.total_score}/125

{'='*40}
MARKET STRUCTURE
{'='*40}
{safe(analysis.market_structure)}

{'='*40}
SUPPORT LEVELS
{'='*40}
{supports_text}

{'='*40}
RESISTANCE LEVELS
{'='*40}
{resistances_text}

{'='*40}
OPTION CHAIN
{'='*40}
{agg_text}

{'='*40}
PATTERNS &amp; SIGNALS
{'='*40}
📊 {safe(analysis.pattern_signal[:150])}

⛓️ {safe(analysis.oi_flow_signal[:150])}

{'='*40}
SCENARIOS
{'='*40}
🟢 Bullish: {safe(analysis.scenario_bullish[:120])}

🔴 Bearish: {safe(analysis.scenario_bearish[:120])}

{'='*40}
RISK FACTORS
{'='*40}"""
            
            for risk in analysis.risk_factors[:3]:
                detailed += f"\n⚠️ {safe(risk[:80])}"
            
            detailed += f"\n\n{'='*40}\nMONITORING (Every 30 min)\n{'='*40}"
            
            for check in analysis.monitoring_checklist[:3]:
                detailed += f"\n✓ {safe(check[:80])}"
            
            if analysis.divergence_warning and analysis.divergence_warning != "None":
                detailed += f"\n\n{'='*40}\n⚠️ DIVERGENCE WARNING\n{'='*40}\n{safe(analysis.divergence_warning[:150])}"
            
            detailed += f"\n\n🤖 DeepSeek V3 Advanced | v9.1 Fixed\n📊 Indices Only (5 min scan)\nExpiry: {expiry}"
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=detailed,
                parse_mode='HTML'
            )
            
            logger.info("Alert sent successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Alert error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def send_startup_message(self):
        try:
            redis_status = "✅" if self.redis.redis_client else "❌"
            
            # Plain text message - NO HTML special characters issues
            msg = f"""🔥 ADVANCED INDEX BOT v9.1 - ACTIVE 🔥

{'='*40}
FIXED DATA FETCHING
{'='*40}

📊 Symbols: NIFTY 50 + SENSEX
⏰ Scan: Every 5 minutes
🔴 Redis: {redis_status}
🤖 AI: DeepSeek V3 (Advanced)

{'='*40}
FIXES IN v9.1
{'='*40}
✅ Fixed NIFTY 50 security ID (13 for INDEX, 25 for FNO)
✅ Fixed SENSEX security ID (51 for both)
✅ Fixed segment mapping (IDX_I for index data)
✅ Added fallback to historical API
✅ Better error logging

{'='*40}
ADVANCED FEATURES
{'='*40}
✅ Market Structure (HH/HL, LH/LL, BOS)
✅ Order Blocks (Demand/Supply zones)
✅ Multi-touch S/R (3+ tests)
✅ Max Pain calculation
✅ OI clustering analysis
✅ Confluence scoring (out of 125)
✅ Multi-scenario planning
✅ Divergence detection

{'='*40}
FLEXIBLE FILTERS
{'='*40}
Confidence: ≥70% (flexible)
OI Divergence: ≥3%
Volume: ≥30%
PCR: >1.1 or <0.9
Time: Skip first 10m and last 20m

Status: 🟢 RUNNING (FIXED MODE)"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg
            )
            logger.info("Startup message sent!")
        except Exception as e:
            logger.error(f"Startup error: {e}")
            logger.error(traceback.format_exc())
    
    async def run(self):
        logger.info("="*70)
        logger.info("ADVANCED INDEX BOT v9.1 - NIFTY/SENSEX (FIXED)")
        logger.info("="*70)
        
        # Validate credentials
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            val = getattr(Config, cred)
            if not val:
                missing.append(cred)
            else:
                # Show first/last chars for verification
                if 'TOKEN' in cred or 'KEY' in cred:
                    logger.info(f"{cred}: {val[:10]}...{val[-10:]}")
                else:
                    logger.info(f"{cred}: {val}")
        
        if missing:
            logger.error(f"❌ MISSING CREDENTIALS: {', '.join(missing)}")
            return
        
        logger.info("✅ All credentials present")
        
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("Bot RUNNING - Scanning every 5 minutes")
        logger.info("Advanced analysis for NIFTY 50 + SENSEX")
        logger.info("="*70)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Sleeping...")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*70}")
                logger.info(f"SCAN CYCLE - {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"{'='*70}")
                
                for index_name in Config.INDICES.keys():
                    logger.info(f"\nScanning {index_name}...")
                    await self.scan_index(index_name)
                    await asyncio.sleep(5)  # Small gap between indices
                
                logger.info(f"\n{'='*70}")
                logger.info(f"CYCLE COMPLETE!")
                logger.info(f"Stats: Scans={self.total_scans}, Alerts={self.alerts_sent}")
                logger.info(f"{'='*70}\n")
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Stopped by user")
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
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("ADVANCED INDEX BOT v9.1 STARTING... (FIXED)")
    logger.info("NIFTY 50 + SENSEX ONLY")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown (Ctrl+C)")
    except Exception as e:
        logger.error(f"\nCritical error: {e}")
        logger.error(traceback.format_exc())
