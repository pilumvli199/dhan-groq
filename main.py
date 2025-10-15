import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import csv
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from PIL import Image
import json

# Technical Analysis - Using 'ta' library (Railway compatible)
import ta
from finta import TA as finta_TA

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

DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# Stocks + Indices (add more as needed)
STOCKS_INDICES = {
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "TCS": {"symbol": "TCS", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ"},
    "ITC": {"symbol": "ITC", "segment": "NSE_EQ"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ"},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
}


# ========================
# ADVANCED TECHNICAL ANALYZER
# ========================
class AdvancedTechnicalAnalyzer:
    """Using 'ta' library - Railway compatible"""
    
    @staticmethod
    def prepare_dataframe(candles):
        """Convert candles to DataFrame"""
        try:
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Ensure column names are lowercase for 'ta' library
            df.columns = df.columns.str.lower()
            
            return df
        except Exception as e:
            logger.error(f"DataFrame prep error: {e}")
            return None
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate 20+ indicators using 'ta' library"""
        try:
            if df is None or len(df) < 50:
                return None
            
            indicators = {}
            
            # Make copy to avoid warnings
            df = df.copy()
            
            # ==================
            # TREND INDICATORS
            # ==================
            
            # 1. Moving Averages
            indicators['SMA_20'] = round(ta.trend.sma_indicator(df['close'], window=20).iloc[-1], 2)
            indicators['SMA_50'] = round(ta.trend.sma_indicator(df['close'], window=50).iloc[-1], 2)
            indicators['EMA_9'] = round(ta.trend.ema_indicator(df['close'], window=9).iloc[-1], 2)
            indicators['EMA_21'] = round(ta.trend.ema_indicator(df['close'], window=21).iloc[-1], 2)
            
            # 2. MACD
            macd = ta.trend.MACD(df['close'])
            indicators['MACD'] = round(macd.macd().iloc[-1], 2)
            indicators['MACD_signal'] = round(macd.macd_signal().iloc[-1], 2)
            indicators['MACD_diff'] = round(macd.macd_diff().iloc[-1], 2)
            
            # 3. ADX (Trend Strength)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            indicators['ADX'] = round(adx.adx().iloc[-1], 2)
            
            # 4. Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            indicators['Ichimoku_A'] = round(ichimoku.ichimoku_a().iloc[-1], 2)
            indicators['Ichimoku_B'] = round(ichimoku.ichimoku_b().iloc[-1], 2)
            
            # ==================
            # MOMENTUM INDICATORS
            # ==================
            
            # 5. RSI
            indicators['RSI'] = round(ta.momentum.rsi(df['close'], window=14).iloc[-1], 2)
            
            # 6. Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            indicators['Stoch_K'] = round(stoch.stoch().iloc[-1], 2)
            indicators['Stoch_D'] = round(stoch.stoch_signal().iloc[-1], 2)
            
            # 7. Williams %R
            indicators['Williams_R'] = round(ta.momentum.williams_r(df['high'], df['low'], df['close']).iloc[-1], 2)
            
            # 8. ROC (Rate of Change)
            indicators['ROC'] = round(ta.momentum.roc(df['close'], window=10).iloc[-1], 2)
            
            # 9. TSI (True Strength Index)
            indicators['TSI'] = round(ta.momentum.tsi(df['close']).iloc[-1], 2)
            
            # ==================
            # VOLATILITY INDICATORS
            # ==================
            
            # 10. Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            indicators['BB_Upper'] = round(bb.bollinger_hband().iloc[-1], 2)
            indicators['BB_Middle'] = round(bb.bollinger_mavg().iloc[-1], 2)
            indicators['BB_Lower'] = round(bb.bollinger_lband().iloc[-1], 2)
            indicators['BB_Width'] = round(bb.bollinger_wband().iloc[-1], 2)
            
            # 11. ATR
            indicators['ATR'] = round(ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1], 2)
            
            # 12. Keltner Channel
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            indicators['KC_Upper'] = round(kc.keltner_channel_hband().iloc[-1], 2)
            indicators['KC_Lower'] = round(kc.keltner_channel_lband().iloc[-1], 2)
            
            # 13. Donchian Channel
            dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
            indicators['DC_Upper'] = round(dc.donchian_channel_hband().iloc[-1], 2)
            indicators['DC_Lower'] = round(dc.donchian_channel_lband().iloc[-1], 2)
            
            # ==================
            # VOLUME INDICATORS
            # ==================
            
            # 14. OBV
            indicators['OBV'] = int(ta.volume.on_balance_volume(df['close'], df['volume']).iloc[-1])
            
            # 15. CMF (Chaikin Money Flow)
            indicators['CMF'] = round(ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume']).iloc[-1], 4)
            
            # 16. MFI (Money Flow Index)
            indicators['MFI'] = round(ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume']).iloc[-1], 2)
            
            # 17. Force Index
            indicators['Force_Index'] = round(ta.volume.force_index(df['close'], df['volume']).iloc[-1], 2)
            
            # ==================
            # CUSTOM CALCULATIONS
            # ==================
            
            # 18. Support & Resistance
            highs = df['high'].tail(50)
            lows = df['low'].tail(50)
            indicators['Resistance'] = round(highs.max(), 2)
            indicators['Support'] = round(lows.min(), 2)
            
            # 19. Volume Analysis
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            indicators['Avg_Volume'] = int(avg_volume)
            indicators['Current_Volume'] = int(current_volume)
            indicators['Volume_Ratio'] = round(current_volume / avg_volume, 2) if avg_volume > 0 else 0
            
            # 20. Price Change
            current_price = df['close'].iloc[-1]
            price_10_back = df['close'].iloc[-10]
            indicators['Price_Change_10'] = round(((current_price - price_10_back) / price_10_back) * 100, 2)
            
            # 21. Trend Direction
            if indicators['SMA_20'] > indicators['SMA_50']:
                indicators['Trend'] = "BULLISH"
            elif indicators['SMA_20'] < indicators['SMA_50']:
                indicators['Trend'] = "BEARISH"
            else:
                indicators['Trend'] = "SIDEWAYS"
            
            # 22. VWAP (from finta)
            try:
                df_finta = df.copy()
                df_finta.columns = [c.capitalize() for c in df_finta.columns]
                vwap = finta_TA.VWAP(df_finta)
                indicators['VWAP'] = round(vwap.iloc[-1], 2) if not vwap.empty else 0
            except:
                indicators['VWAP'] = 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def generate_signals(indicators, current_price):
        """Generate trading signals based on indicators"""
        try:
            if not indicators:
                return None
            
            signal = {
                'type': None,
                'strength': 0,
                'entry': None,
                'target1': None,
                'target2': None,
                'sl': None,
                'confidence': 0,
                'reasons': [],
                'risk_reward': 0
            }
            
            bullish_score = 0
            bearish_score = 0
            
            # ==================
            # TREND ANALYSIS
            # ==================
            
            # MA Alignment
            if current_price > indicators['EMA_9'] > indicators['EMA_21'] > indicators['SMA_20']:
                bullish_score += 15
                signal['reasons'].append("✓ Strong MA alignment (bullish)")
            elif current_price < indicators['EMA_9'] < indicators['EMA_21'] < indicators['SMA_20']:
                bearish_score += 15
                signal['reasons'].append("✓ Strong MA alignment (bearish)")
            
            # ADX
            adx = indicators.get('ADX', 0)
            if adx > 25:
                if indicators['Trend'] == "BULLISH":
                    bullish_score += 10
                    signal['reasons'].append(f"✓ Strong bullish trend (ADX: {adx})")
                else:
                    bearish_score += 10
                    signal['reasons'].append(f"✓ Strong bearish trend (ADX: {adx})")
            
            # MACD
            macd = indicators.get('MACD', 0)
            macd_signal_line = indicators.get('MACD_signal', 0)
            if macd > macd_signal_line and macd > 0:
                bullish_score += 10
                signal['reasons'].append("✓ MACD bullish crossover")
            elif macd < macd_signal_line and macd < 0:
                bearish_score += 10
                signal['reasons'].append("✓ MACD bearish crossover")
            
            # ==================
            # MOMENTUM
            # ==================
            
            # RSI
            rsi = indicators.get('RSI', 50)
            if 40 < rsi < 70:
                bullish_score += 8
                signal['reasons'].append(f"✓ RSI bullish zone ({rsi:.1f})")
            elif 30 < rsi < 60:
                bearish_score += 8
                signal['reasons'].append(f"✓ RSI bearish zone ({rsi:.1f})")
            
            # Stochastic
            stoch_k = indicators.get('Stoch_K', 50)
            stoch_d = indicators.get('Stoch_D', 50)
            if stoch_k > stoch_d and stoch_k < 80:
                bullish_score += 7
                signal['reasons'].append("✓ Stochastic bullish")
            elif stoch_k < stoch_d and stoch_k > 20:
                bearish_score += 7
                signal['reasons'].append("✓ Stochastic bearish")
            
            # MFI
            mfi = indicators.get('MFI', 50)
            if mfi > 50 and mfi < 80:
                bullish_score += 6
                signal['reasons'].append(f"✓ Money flowing in (MFI: {mfi:.1f})")
            elif mfi < 50 and mfi > 20:
                bearish_score += 6
                signal['reasons'].append(f"✓ Money flowing out (MFI: {mfi:.1f})")
            
            # ==================
            # VOLATILITY
            # ==================
            
            # Bollinger Bands
            bb_upper = indicators.get('BB_Upper', 0)
            bb_lower = indicators.get('BB_Lower', 0)
            
            if current_price > bb_upper:
                bullish_score += 8
                signal['reasons'].append("✓ Above Bollinger upper band")
            elif current_price < bb_lower:
                bearish_score += 8
                signal['reasons'].append("✓ Below Bollinger lower band")
            
            # Support/Resistance
            resistance = indicators.get('Resistance', 0)
            support = indicators.get('Support', 0)
            
            if current_price > (resistance * 0.99):
                bullish_score += 10
                signal['reasons'].append(f"✓ Near resistance breakout (₹{resistance})")
            elif current_price < (support * 1.01):
                bearish_score += 10
                signal['reasons'].append(f"✓ Near support breakdown (₹{support})")
            
            # ==================
            # VOLUME
            # ==================
            
            volume_ratio = indicators.get('Volume_Ratio', 1)
            if volume_ratio > 1.5:
                if bullish_score > bearish_score:
                    bullish_score += 10
                    signal['reasons'].append(f"✓ Volume spike {volume_ratio:.1f}x (bullish)")
                else:
                    bearish_score += 10
                    signal['reasons'].append(f"✓ Volume spike {volume_ratio:.1f}x (bearish)")
            
            # ==================
            # SIGNAL GENERATION
            # ==================
            
            if bullish_score > bearish_score and bullish_score >= 50:
                signal['type'] = "BULLISH"
                signal['strength'] = min(100, bullish_score)
                signal['confidence'] = int((bullish_score / (bullish_score + bearish_score)) * 100)
                
                atr = indicators.get('ATR', 0)
                signal['entry'] = current_price
                signal['target1'] = current_price + (atr * 1.5) if atr > 0 else current_price * 1.02
                signal['target2'] = current_price + (atr * 3) if atr > 0 else current_price * 1.04
                signal['sl'] = max(support, current_price - atr) if atr > 0 else support
                
            elif bearish_score > bullish_score and bearish_score >= 50:
                signal['type'] = "BEARISH"
                signal['strength'] = min(100, bearish_score)
                signal['confidence'] = int((bearish_score / (bullish_score + bearish_score)) * 100)
                
                atr = indicators.get('ATR', 0)
                signal['entry'] = current_price
                signal['target1'] = current_price - (atr * 1.5) if atr > 0 else current_price * 0.98
                signal['target2'] = current_price - (atr * 3) if atr > 0 else current_price * 0.96
                signal['sl'] = min(resistance, current_price + atr) if atr > 0 else resistance
                
            else:
                return None
            
            # Calculate R:R
            if signal['sl'] and signal['target1']:
                risk = abs(signal['entry'] - signal['sl'])
                reward = abs(signal['target1'] - signal['entry'])
                signal['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
            
            # Filters
            if signal['confidence'] < 70:
                return None
            
            if signal['risk_reward'] < 1.5:
                return None
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None


# ========================
# TRADING BOT
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
        self.tech_analyzer = AdvancedTechnicalAnalyzer()
        
        logger.info("🤖 Trading Bot initialized (Railway compatible)")
    
    async def load_security_ids(self):
        """Load security IDs"""
        try:
            logger.info("Loading security IDs...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code == 200:
                csv_data = response.text.split('\n')
                
                for symbol, info in STOCKS_INDICES.items():
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
                
                logger.info(f"✅ Loaded {len(self.security_id_map)} symbols")
                return True
            return False
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Fetch candles"""
        try:
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=10)
            
            payload = {
                "securityId": st
