"""
🤖 HYBRID NIFTY 50 STOCKS TRADING BOT v10.0
Version: 10.0 - TWO-STEP FILTER (HYBRID APPROACH)
Phase 1: Quick scan all 50 stocks (5 sec each)
Phase 2: Deep analysis on promising stocks (30 sec each)
Scan Interval: 15 minutes
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
from typing import Dict, List, Optional, Tuple
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
    """Bot Configuration - HYBRID TWO-STEP APPROACH"""
    
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
    DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
    
    SCAN_INTERVAL = 900  # 15 minutes
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 86400
    
    # PHASE 1: Quick Filter (Lenient)
    PHASE1_CONFIDENCE_MIN = 70
    PHASE1_OI_DIVERGENCE_MIN = 2.5
    PHASE1_VOLUME_MIN = 25.0
    
    # PHASE 2: Deep Analysis (Stricter)
    PHASE2_CONFIDENCE_MIN = 75
    PHASE2_SCORE_MIN = 90  # Out of 125
    PHASE2_ALIGNMENT_MIN = 18  # Out of 25
    
    SKIP_OPENING_MINUTES = 10
    SKIP_CLOSING_MINUTES = 20
    
    LOOKBACK_DAYS = 10
    ATM_STRIKE_RANGE = 11
    MIN_CANDLES_REQUIRED = 50
    
    NIFTY_50_STOCKS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
        "ASIANPAINT", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK",
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
        "DMART", "WIPRO", "BAJAJFINSV", "ADANIENT", "ONGC",
        "NTPC", "TECHM", "POWERGRID", "M&M", "TATASTEEL",
        "INDUSINDBK", "COALINDIA", "JSWSTEEL", "GRASIM", "BRITANNIA",
        "TATACONSUM", "HINDALCO", "EICHERMOT", "ADANIPORTS", "APOLLOHOSP",
        "SBILIFE", "BAJAJ-AUTO", "CIPLA", "DIVISLAB", "HDFCLIFE",
        "BPCL", "HEROMOTOCO", "TATAMOTORS", "UPL", "DRREDDY"
    ]


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
    ce_oi_change_pct: float
    pe_oi_change_pct: float
    ce_volume_change_pct: float
    pe_volume_change_pct: float
    pcr: float
    overall_sentiment: str
    max_pain: float = 0.0


@dataclass
class QuickAnalysis:
    """Phase 1: Quick scan result"""
    opportunity: str
    confidence: int
    oi_divergence: float
    volume_surge: float
    pcr: float
    passed_phase1: bool
    reason: str


@dataclass
class DeepAnalysis:
    """Phase 2: Deep analysis result"""
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
            
            aggregate_analysis = AggregateOIAnalysis(
                total_ce_oi=total_ce_oi_new,
                total_pe_oi=total_pe_oi_new,
                total_ce_volume=total_ce_volume_new,
                total_pe_volume=total_pe_volume_new,
                ce_oi_change_pct=ce_oi_change_pct,
                pe_oi_change_pct=pe_oi_change_pct,
                ce_volume_change_pct=ce_volume_change_pct,
                pe_volume_change_pct=pe_volume_change_pct,
                pcr=pcr,
                overall_sentiment=sentiment
            )
            
            return {
                'change': 'UPDATED',
                'aggregate_analysis': aggregate_analysis
            }
            
        except Exception as e:
            logger.error(f"Redis comparison error: {e}")
            return {'change': 'ERROR', 'aggregate_analysis': None}


class AdvancedChartAnalyzer:
    @staticmethod
    def identify_market_structure(df: pd.DataFrame) -> Dict:
        try:
            if len(df) < 20:
                return {"structure": "INSUFFICIENT", "bias": "NEUTRAL"}
            
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
                if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
                    return {"structure": "HH_HL", "bias": "BULLISH"}
                elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
                    return {"structure": "LH_LL", "bias": "BEARISH"}
            
            return {"structure": "SIDEWAYS", "bias": "NEUTRAL"}
        
        except:
            return {"structure": "ERROR", "bias": "NEUTRAL"}
    
    @staticmethod
    def calculate_multi_touch_sr(df: pd.DataFrame) -> Dict:
        try:
            if len(df) < 50:
                current = df['close'].iloc[-1]
                return {
                    'supports': [current * 0.98],
                    'resistances': [current * 1.02]
                }
            
            recent = df.tail(100)
            current = recent['close'].iloc[-1]
            
            highs = recent['high'].values
            lows = recent['low'].values
            
            resistance_levels = []
            support_levels = []
            
            window = 5
            for i in range(window, len(recent) - window):
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(highs[i])
                
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    support_levels.append(lows[i])
            
            def cluster(levels):
                if not levels:
                    return []
                levels = sorted(levels)
                clustered = []
                current_cluster = [levels[0]]
                for level in levels[1:]:
                    if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.005:
                        current_cluster.append(level)
                    else:
                        clustered.append(np.mean(current_cluster))
                        current_cluster = [level]
                clustered.append(np.mean(current_cluster))
                return clustered
            
            resistances = cluster(resistance_levels)
            supports = cluster(support_levels)
            
            resistances = [r for r in resistances if 0.001 <= (r - current)/current <= 0.05]
            supports = [s for s in supports if 0.001 <= (current - s)/current <= 0.05]
            
            return {
                'supports': supports[:3] if supports else [current * 0.98],
                'resistances': resistances[:3] if resistances else [current * 1.02]
            }
        
        except:
            current = df['close'].iloc[-1]
            return {
                'supports': [current * 0.98],
                'resistances': [current * 1.02]
            }


class DhanAPI:
    def __init__(self, redis_cache: RedisCache):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json'
        }
        self.security_id_map = {}
        self.redis = redis_cache
        logger.info("DhanAPI initialized")
    
    async def load_security_ids(self):
        try:
            logger.info("Loading NIFTY 50 stock security IDs...")
            response = requests.get(Config.DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code != 200:
                return False
            
            csv_reader = csv.DictReader(io.StringIO(response.text))
            all_rows = list(csv_reader)
            
            for stock_symbol in Config.NIFTY_50_STOCKS:
                for row in all_rows:
                    try:
                        trading_symbol = row.get('SEM_TRADING_SYMBOL', '').strip()
                        segment = row.get('SEM_SEGMENT', '').strip()
                        exch_segment = row.get('SEM_EXM_EXCH_ID', '').strip()
                        
                        if (segment == 'E' and exch_segment == 'NSE' and trading_symbol == stock_symbol):
                            sec_id = row.get('SEM_SMST_SECURITY_ID', '').strip()
                            if sec_id:
                                self.security_id_map[stock_symbol] = {
                                    'security_id': int(sec_id),
                                    'segment': 'NSE_EQ',
                                    'trading_symbol': trading_symbol
                                }
                                logger.info(f"{stock_symbol}: ID={sec_id}")
                                break
                    except Exception:
                        continue
            
            logger.info(f"Loaded {len(self.security_id_map)}/50 stocks")
            return len(self.security_id_map) > 0
            
        except Exception as e:
            logger.error(f"Error loading securities: {e}")
            return False
    
    def get_nearest_expiry(self, security_id: int) -> Optional[str]:
        try:
            payload = {
                "UnderlyingScrip": int(security_id),
                "UnderlyingSeg": "NSE_EQ"
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
                    return data['data'][0]
            return None
            
        except:
            return None
    
    def get_chart_data(self, security_id: int, symbol: str) -> Optional[pd.DataFrame]:
        try:
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=Config.LOOKBACK_DAYS)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "expiryCode": 0,
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
                return None
            
            data = response.json()
            
            if 'timestamp' not in data or len(data['open']) == 0:
                return None
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            })
            
            df = df.dropna()
            df.set_index('timestamp', inplace=True)
            
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return df_15m
            
        except:
            return None
    
    def get_option_chain(self, security_id: int, expiry: str, spot_price: float) -> Optional[List[OIData]]:
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": "NSE_EQ",
                "Expiry": expiry
            }
            
            response = requests.post(
                Config.DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('data'):
                return None
            
            oc_data = data['data'].get('oc', {})
            
            strikes = [float(s) for s in oc_data.keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            oi_list = []
            
            for strike_str, strike_data in oc_data.items():
                try:
                    strike = float(strike_str)
                    
                    if abs(strike - atm_strike) > (atm_strike * 0.05):
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
                except:
                    continue
            
            return oi_list
            
        except:
            return None


class QuickScanner:
    """Phase 1: Quick scan with simple prompt"""
    
    @staticmethod
    def quick_analysis(symbol: str, spot_price: float, aggregate: AggregateOIAnalysis) -> Optional[QuickAnalysis]:
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Quick analysis for {symbol} options trading.

Spot: {spot_price:.2f}
PCR: {aggregate.pcr:.2f}
CE OI Change: {aggregate.ce_oi_change_pct:+.2f}%
PE OI Change: {aggregate.pe_oi_change_pct:+.2f}%
CE Volume: {aggregate.ce_volume_change_pct:+.2f}%
PE Volume: {aggregate.pe_volume_change_pct:+.2f}%
Sentiment: {aggregate.overall_sentiment}

Reply JSON only:
{{
  "opportunity": "PE_BUY or CE_BUY or WAIT",
  "confidence": 75,
  "reason": "Brief reason"
}}"""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Quick trader. Reply JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 300
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            analysis_dict = QuickScanner.extract_json(content)
            
            if not analysis_dict:
                return None
            
            opportunity = analysis_dict.get('opportunity', 'WAIT')
            confidence = analysis_dict.get('confidence', 0)
            
            oi_divergence = abs(aggregate.pe_oi_change_pct - aggregate.ce_oi_change_pct)
            
            if opportunity == "PE_BUY":
                volume_surge = aggregate.pe_volume_change_pct
            elif opportunity == "CE_BUY":
                volume_surge = aggregate.ce_volume_change_pct
            else:
                volume_surge = 0
            
            passed = (
                confidence >= Config.PHASE1_CONFIDENCE_MIN and
                oi_divergence >= Config.PHASE1_OI_DIVERGENCE_MIN and
                volume_surge >= Config.PHASE1_VOLUME_MIN and
                opportunity != "WAIT"
            )
            
            return QuickAnalysis(
                opportunity=opportunity,
                confidence=confidence,
                oi_divergence=oi_divergence,
                volume_surge=volume_surge,
                pcr=aggregate.pcr,
                passed_phase1=passed,
                reason=analysis_dict.get('reason', 'N/A')
            )
            
        except Exception as e:
            logger.error(f"Quick analysis error: {e}")
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
        except:
            return None


class DeepAnalyzer:
    """Phase 2: Deep analysis with advanced prompt"""
    
    @staticmethod
    def deep_analysis(symbol: str, spot_price: float, df: pd.DataFrame,
                     aggregate: AggregateOIAnalysis, structure: Dict, sr_levels: Dict) -> Optional[DeepAnalysis]:
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""DEEP analysis for {symbol} F&O trading.

Spot: {spot_price:.2f}

STRUCTURE: {structure['structure']} | {structure['bias']}

SUPPORT: {', '.join([f"{s:.0f}" for s in sr_levels['supports'][:3]])}
RESISTANCE: {', '.join([f"{r:.0f}" for r in sr_levels['resistances'][:3]])}

OPTIONS:
PCR: {aggregate.pcr:.2f}
CE: {aggregate.ce_oi_change_pct:+.2f}% | Vol: {aggregate.ce_volume_change_pct:+.2f}%
PE: {aggregate.pe_oi_change_pct:+.2f}% | Vol: {aggregate.pe_volume_change_pct:+.2f}%

Score out of 125:
- Chart: /50
- Options: /50
- Alignment: /25

Reply JSON:
{{
  "opportunity": "PE_BUY or CE_BUY",
  "confidence": 78,
  "chart_score": 40,
  "option_score": 42,
  "alignment_score": 20,
  "total_score": 102,
  "entry_price": {spot_price:.2f},
  "stop_loss": {spot_price * 0.995:.2f},
  "target_1": {spot_price * 1.01:.2f},
  "target_2": {spot_price * 1.02:.2f},
  "risk_reward": "1:2",
  "recommended_strike": {int(spot_price)},
  "pattern_signal": "Pattern",
  "oi_flow_signal": "OI flow",
  "market_structure": "{structure['structure']}",
  "support_levels": {sr_levels['supports'][:2]},
  "resistance_levels": {sr_levels['resistances'][:2]},
  "scenario_bullish": "If breaks X",
  "scenario_bearish": "If breaks Y",
  "risk_factors": ["Risk1", "Risk2"],
  "monitoring_checklist": ["Check1", "Check2"]
}}"""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Expert trader. Reply JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            
            if response.status_code != 200:
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            analysis_dict = DeepAnalyzer.extract_json(content)
            
            if not analysis_dict:
                return None
            
            required = ['opportunity', 'confidence', 'chart_score', 'option_score', 'alignment_score']
            if not all(f in analysis_dict for f in required):
                return None
            
            return DeepAnalysis(
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
                scenario_bullish=analysis_dict.get('scenario_bullish', 'N/A'),
                scenario_bearish=analysis_dict.get('scenario_bearish', 'N/A'),
                risk_factors=analysis_dict.get('risk_factors', ['See analysis']),
                monitoring_checklist=analysis_dict.get('monitoring_checklist', ['Monitor price'])
            )
            
        except Exception as e:
            logger.error(f"Deep analysis error: {e}")
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
        except:
            return None


class ChartGenerator:
    @staticmethod
    def create_chart(df: pd.DataFrame, symbol: str, analysis: DeepAnalysis) -> Optional[BytesIO]:
        try:
            chart_df = df.tail(100).copy()
            
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
            
            ax.text(len(chart_df), analysis.entry_price, f' Entry: {analysis.entry_price:.1f}', 
                   color='blue', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), analysis.target_1, f' T1: {analysis.target_1:.1f}', 
                   color='green', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), analysis.target_2, f' T2: {analysis.target_2:.1f}', 
                   color='darkgreen', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), analysis.stop_loss, f' SL: {analysis.stop_loss:.1f}', 
                   color='red', fontweight='bold', va='center', fontsize=10)
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None


class HybridNifty50Bot:
    def __init__(self):
        logger.info("Initializing Hybrid NIFTY 50 Bot v10.0...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.quick_scanner = QuickScanner()
        self.deep_analyzer = DeepAnalyzer()
        self.chart_analyzer = AdvancedChartAnalyzer()
        self.chart_gen = ChartGenerator()
        self.running = True
        
        self.phase1_scanned = 0
        self.phase1_passed = 0
        self.phase2_analyzed = 0
        self.alerts_sent = 0
        
        logger.info("Bot v10.0 initialized - HYBRID MODE")
    
    def is_market_open(self) -> bool:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        return Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE
    
    def escape_html(self, text: str) -> str:
        return html.escape(str(text))
    
    async def phase1_quick_scan(self) -> List[Tuple[str, Dict, QuickAnalysis, AggregateOIAnalysis]]:
        """Phase 1: Quick scan all 50 stocks"""
        promising_stocks = []
        
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: QUICK SCAN (All 50 stocks)")
        logger.info("="*70)
        
        for idx, (symbol, info) in enumerate(self.dhan.security_id_map.items(), 1):
            try:
                self.phase1_scanned += 1
                
                logger.info(f"[{idx}/50] Quick scan: {symbol}")
                
                security_id = info['security_id']
                
                expiry = self.dhan.get_nearest_expiry(security_id)
                if not expiry:
                    continue
                
                df = self.dhan.get_chart_data(security_id, symbol)
                if df is None or len(df) < 30:
                    continue
                
                spot_price = df['close'].iloc[-1]
                
                oi_data = self.dhan.get_option_chain(security_id, expiry, spot_price)
                if not oi_data or len(oi_data) < 10:
                    continue
                
                oi_comparison = self.redis.get_oi_comparison(symbol, oi_data, spot_price)
                self.redis.store_option_chain(symbol, oi_data, spot_price)
                
                aggregate = oi_comparison.get('aggregate_analysis')
                if not aggregate:
                    continue
                
                quick = self.quick_scanner.quick_analysis(symbol, spot_price, aggregate)
                
                if quick and quick.passed_phase1:
                    self.phase1_passed += 1
                    promising_stocks.append((symbol, info, quick, aggregate))
                    logger.info(f"✅ {symbol}: PASSED Phase 1 (Conf: {quick.confidence}%, Div: {quick.oi_divergence:.1f}%, Vol: {quick.volume_surge:.1f}%)")
                else:
                    logger.info(f"❌ {symbol}: Failed Phase 1")
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Phase 1 error {symbol}: {e}")
        
        logger.info(f"\nPhase 1 Complete: {self.phase1_passed}/{self.phase1_scanned} stocks passed")
        
        return promising_stocks
    
    async def phase2_deep_analysis(self, promising_stocks: List[Tuple[str, Dict, QuickAnalysis, AggregateOIAnalysis]]):
        """Phase 2: Deep analysis on promising stocks"""
        
        if not promising_stocks:
            logger.info("No stocks passed Phase 1 - skipping Phase 2")
            return
        
        logger.info("\n" + "="*70)
        logger.info(f"PHASE 2: DEEP ANALYSIS ({len(promising_stocks)} promising stocks)")
        logger.info("="*70)
        
        for idx, (symbol, info, quick, aggregate) in enumerate(promising_stocks, 1):
            try:
                self.phase2_analyzed += 1
                
                logger.info(f"\n[{idx}/{len(promising_stocks)}] Deep analysis: {symbol}")
                
                security_id = info['security_id']
                
                df = self.dhan.get_chart_data(security_id, symbol)
                if df is None or len(df) < 50:
                    logger.warning(f"{symbol}: Insufficient chart data")
                    continue
                
                spot_price = df['close'].iloc[-1]
                
                structure = self.chart_analyzer.identify_market_structure(df)
                sr_levels = self.chart_analyzer.calculate_multi_touch_sr(df)
                
                logger.info(f"{symbol}: Structure={structure['structure']}, Bias={structure['bias']}")
                
                deep = self.deep_analyzer.deep_analysis(symbol, spot_price, df, aggregate, structure, sr_levels)
                
                if not deep:
                    logger.warning(f"{symbol}: No deep analysis")
                    continue
                
                logger.info(f"{symbol}: Score={deep.total_score}/125 (Chart:{deep.chart_score} Opt:{deep.option_score} Align:{deep.alignment_score})")
                
                if deep.confidence < Config.PHASE2_CONFIDENCE_MIN:
                    logger.info(f"❌ {symbol}: Confidence {deep.confidence}% < {Config.PHASE2_CONFIDENCE_MIN}%")
                    continue
                
                if deep.total_score < Config.PHASE2_SCORE_MIN:
                    logger.info(f"❌ {symbol}: Score {deep.total_score} < {Config.PHASE2_SCORE_MIN}")
                    continue
                
                if deep.alignment_score < Config.PHASE2_ALIGNMENT_MIN:
                    logger.info(f"❌ {symbol}: Alignment {deep.alignment_score} < {Config.PHASE2_ALIGNMENT_MIN}")
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                now_ist = datetime.now(ist)
                hour = now_ist.hour
                minute = now_ist.minute
                
                if hour == 9 and minute < 15 + Config.SKIP_OPENING_MINUTES:
                    logger.info(f"❌ {symbol}: Market opening period")
                    continue
                
                if hour == 15 or (hour == 14 and minute >= (60 - Config.SKIP_CLOSING_MINUTES)):
                    logger.info(f"❌ {symbol}: Market closing period")
                    continue
                
                logger.info(f"✅ {symbol}: PASSED Phase 2 - Generating alert!")
                
                expiry = self.dhan.get_nearest_expiry(security_id)
                
                chart_image = self.chart_gen.create_chart(df, symbol, deep)
                
                await self.send_alert(symbol, spot_price, deep, aggregate, expiry, chart_image)
                
                self.alerts_sent += 1
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Phase 2 error {symbol}: {e}")
                logger.error(traceback.format_exc())
    
    async def send_alert(self, symbol: str, spot_price: float, analysis: DeepAnalysis,
                        aggregate: AggregateOIAnalysis, expiry: str, chart_image: Optional[BytesIO]):
        try:
            signal_map = {
                "PE_BUY": ("🔴", "PE BUY"),
                "CE_BUY": ("🟢", "CE BUY")
            }
            
            signal_emoji, signal_text = signal_map.get(analysis.opportunity, ("⚪", "WAIT"))
            
            def safe(val):
                return self.escape_html(val)
            
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
            
            caption = f"🎯 HYBRID SIGNAL - {safe(symbol)}\n\n{signal_emoji} {signal_text} | Confidence: {analysis.confidence}%\nScore: {analysis.total_score}/125 (C:{analysis.chart_score} O:{analysis.option_score} A:{analysis.alignment_score})\n\n💰 Entry: {analysis.entry_price:.1f} | SL: {analysis.stop_loss:.1f}\n🎯 T1: {analysis.target_1:.1f} | T2: {analysis.target_2:.1f}\nRR: {analysis.risk_reward} | Strike: {analysis.recommended_strike}\n\n⏰ {ist_time} IST | v10.0 Hybrid"
            
            if chart_image:
                try:
                    await self.bot.send_photo(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        photo=chart_image,
                        caption=caption,
                        parse_mode='HTML'
                    )
                except:
                    await self.bot.send_message(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        text=caption,
                        parse_mode='HTML'
                    )
            
            supports_text = ", ".join([f"{s:.1f}" for s in analysis.support_levels[:2]])
            resistances_text = ", ".join([f"{r:.1f}" for r in analysis.resistance_levels[:2]])
            
            detailed = f"""🎯 HYBRID ANALYSIS - {safe(symbol)}

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

Support: {supports_text}
Resistance: {resistances_text}

{'='*40}
OPTIONS
{'='*40}
PCR: {aggregate.pcr:.2f}
CE: {aggregate.ce_oi_change_pct:+.1f}% | Vol: {aggregate.ce_volume_change_pct:+.1f}%
PE: {aggregate.pe_oi_change_pct:+.1f}% | Vol: {aggregate.pe_volume_change_pct:+.1f}%

{'='*40}
SIGNALS
{'='*40}
📊 {safe(analysis.pattern_signal[:100])}
⛓️ {safe(analysis.oi_flow_signal[:100])}

{'='*40}
SCENARIOS
{'='*40}
🟢 {safe(analysis.scenario_bullish[:100])}
🔴 {safe(analysis.scenario_bearish[:100])}

{'='*40}
RISKS
{'='*40}"""
            
            for risk in analysis.risk_factors[:3]:
                detailed += f"\n⚠️ {safe(risk[:70])}"
            
            detailed += f"\n\n{'='*40}\nMONITOR\n{'='*40}"
            
            for check in analysis.monitoring_checklist[:3]:
                detailed += f"\n✓ {safe(check[:70])}"
            
            detailed += f"\n\n🤖 DeepSeek V3 Hybrid | v10.0\nExpiry: {expiry}\n2-Step Filter: Quick → Deep"
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=detailed,
                parse_mode='HTML'
            )
            
            logger.info("Alert sent successfully!")
            
        except Exception as e:
            logger.error(f"Alert error: {e}")
    
    async def send_startup_message(self):
        try:
            redis_status = "✅" if self.redis.redis_client else "❌"
            
            msg = f"""🔥 HYBRID NIFTY 50 BOT v10.0 - ACTIVE 🔥

{'='*40}
TWO-STEP HYBRID FILTER
{'='*40}

📊 Stocks: All 50 NIFTY stocks
⏰ Scan: Every 15 minutes
🔴 Redis: {redis_status}
🤖 AI: DeepSeek V3 (Hybrid)

{'='*40}
PHASE 1: QUICK SCAN
{'='*40}
✅ All 50 stocks (5 sec each)
✅ Simple analysis
✅ Lenient filters:
   - Confidence: ≥70%
   - OI Divergence: ≥2.5%
   - Volume: ≥25%
✅ Time: ~250 sec (4 min)

{'='*40}
PHASE 2: DEEP ANALYSIS
{'='*40}
✅ Promising stocks only (5-8)
✅ Advanced analysis:
   - Market structure
   - Multi-touch S/R
   - Confluence scoring
✅ Stricter filters:
   - Confidence: ≥75%
   - Score: ≥90/125
   - Alignment: ≥18/25
✅ Time: ~200 sec (3.5 min)

{'='*40}
TOTAL CYCLE TIME
{'='*40}
Phase 1 + Phase 2: ~450 sec (7.5 min)
Buffer: 7.5 minutes
Fits in 15-min interval: ✅

{'='*40}
EXPECTED RESULTS
{'='*40}
Phase 1 pass: 5-8 stocks
Final signals: 2-3 per cycle
Daily signals: 6-9 total
Quality: Premium (double-filtered)
Win Rate Target: 82-87%

Status: 🟢 RUNNING (HYBRID MODE)"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("Startup message sent!")
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    async def run(self):
        logger.info("="*70)
        logger.info("HYBRID NIFTY 50 BOT v10.0")
        logger.info("="*70)
        
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            if not getattr(Config, cred):
                missing.append(cred)
        
        if missing:
            logger.error(f"Missing: {', '.join(missing)}")
            return
        
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("Failed to load securities")
            return
        
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("Bot RUNNING - Hybrid two-step filter")
        logger.info("="*70)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Sleeping...")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*70}")
                logger.info(f"HYBRID SCAN CYCLE - {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"{'='*70}")
                
                promising_stocks = await self.phase1_quick_scan()
                
                await self.phase2_deep_analysis(promising_stocks)
                
                logger.info(f"\n{'='*70}")
                logger.info(f"CYCLE COMPLETE!")
                logger.info(f"Phase 1: {self.phase1_passed}/{self.phase1_scanned} passed")
                logger.info(f"Phase 2: Analyzed {self.phase2_analyzed}")
                logger.info(f"Alerts sent: {self.alerts_sent}")
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
        bot = HybridNifty50Bot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("HYBRID NIFTY 50 BOT v10.0 STARTING...")
    logger.info("Two-Step Filter: Quick → Deep")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown (Ctrl+C)")
    except Exception as e:
        logger.error(f"\nCritical: {e}")
        logger.error(traceback.format_exc())
