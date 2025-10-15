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

# Advanced Technical Analysis Libraries
import pandas_ta as ta
from finta import TA

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

# Stocks + Indices
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
    """
    Using pandas-ta & finta for 130+ indicators
    """
    
    @staticmethod
    def prepare_dataframe(candles):
        """Convert candles to DataFrame"""
        try:
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Rename columns for TA libraries
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"DataFrame prep error: {e}")
            return None
    
    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate 20+ advanced indicators
        Returns comprehensive technical analysis
        """
        try:
            if df is None or len(df) < 50:
                return None
            
            indicators = {}
            
            # ==================
            # TREND INDICATORS
            # ==================
            
            # 1. Moving Averages (pandas-ta)
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.ema(length=9, append=True)
            df.ta.ema(length=21, append=True)
            
            indicators['SMA_20'] = round(df['SMA_20'].iloc[-1], 2)
            indicators['SMA_50'] = round(df['SMA_50'].iloc[-1], 2)
            indicators['EMA_9'] = round(df['EMA_9'].iloc[-1], 2)
            indicators['EMA_21'] = round(df['EMA_21'].iloc[-1], 2)
            
            # 2. MACD (Moving Average Convergence Divergence)
            df.ta.macd(append=True)
            indicators['MACD'] = round(df['MACD_12_26_9'].iloc[-1], 2) if 'MACD_12_26_9' in df.columns else 0
            indicators['MACD_signal'] = round(df['MACDs_12_26_9'].iloc[-1], 2) if 'MACDs_12_26_9' in df.columns else 0
            indicators['MACD_hist'] = round(df['MACDh_12_26_9'].iloc[-1], 2) if 'MACDh_12_26_9' in df.columns else 0
            
            # 3. ADX (Average Directional Index) - Trend Strength
            df.ta.adx(append=True)
            indicators['ADX'] = round(df['ADX_14'].iloc[-1], 2) if 'ADX_14' in df.columns else 0
            
            # 4. SuperTrend
            df.ta.supertrend(append=True)
            indicators['SuperTrend'] = round(df['SUPERT_7_3.0'].iloc[-1], 2) if 'SUPERT_7_3.0' in df.columns else 0
            
            # ==================
            # MOMENTUM INDICATORS
            # ==================
            
            # 5. RSI (Relative Strength Index)
            df.ta.rsi(length=14, append=True)
            indicators['RSI'] = round(df['RSI_14'].iloc[-1], 2) if 'RSI_14' in df.columns else 50
            
            # 6. Stochastic Oscillator
            df.ta.stoch(append=True)
            indicators['Stoch_K'] = round(df['STOCHk_14_3_3'].iloc[-1], 2) if 'STOCHk_14_3_3' in df.columns else 0
            indicators['Stoch_D'] = round(df['STOCHd_14_3_3'].iloc[-1], 2) if 'STOCHd_14_3_3' in df.columns else 0
            
            # 7. Williams %R
            df.ta.willr(append=True)
            indicators['Williams_R'] = round(df['WILLR_14'].iloc[-1], 2) if 'WILLR_14' in df.columns else 0
            
            # 8. CCI (Commodity Channel Index)
            df.ta.cci(append=True)
            indicators['CCI'] = round(df['CCI_14_0.015'].iloc[-1], 2) if 'CCI_14_0.015' in df.columns else 0
            
            # 9. MFI (Money Flow Index)
            df.ta.mfi(append=True)
            indicators['MFI'] = round(df['MFI_14'].iloc[-1], 2) if 'MFI_14' in df.columns else 50
            
            # ==================
            # VOLATILITY INDICATORS
            # ==================
            
            # 10. Bollinger Bands
            df.ta.bbands(append=True)
            indicators['BB_Upper'] = round(df['BBU_5_2.0'].iloc[-1], 2) if 'BBU_5_2.0' in df.columns else 0
            indicators['BB_Middle'] = round(df['BBM_5_2.0'].iloc[-1], 2) if 'BBM_5_2.0' in df.columns else 0
            indicators['BB_Lower'] = round(df['BBL_5_2.0'].iloc[-1], 2) if 'BBL_5_2.0' in df.columns else 0
            
            # 11. ATR (Average True Range)
            df.ta.atr(append=True)
            indicators['ATR'] = round(df['ATR_14'].iloc[-1], 2) if 'ATR_14' in df.columns else 0
            
            # 12. Keltner Channels
            df.ta.kc(append=True)
            indicators['KC_Upper'] = round(df['KCUe_20_2'].iloc[-1], 2) if 'KCUe_20_2' in df.columns else 0
            indicators['KC_Lower'] = round(df['KCLe_20_2'].iloc[-1], 2) if 'KCLe_20_2' in df.columns else 0
            
            # ==================
            # VOLUME INDICATORS
            # ==================
            
            # 13. OBV (On Balance Volume)
            df.ta.obv(append=True)
            indicators['OBV'] = int(df['OBV'].iloc[-1]) if 'OBV' in df.columns else 0
            
            # 14. VWAP (Volume Weighted Average Price)
            df.ta.vwap(append=True)
            indicators['VWAP'] = round(df['VWAP_D'].iloc[-1], 2) if 'VWAP_D' in df.columns else 0
            
            # 15. AD (Accumulation/Distribution)
            df.ta.ad(append=True)
            indicators['AD'] = int(df['AD'].iloc[-1]) if 'AD' in df.columns else 0
            
            # ==================
            # CUSTOM CALCULATIONS
            # ==================
            
            # 16. Support & Resistance
            highs = df['High'].tail(50)
            lows = df['Low'].tail(50)
            indicators['Resistance'] = round(highs.max(), 2)
            indicators['Support'] = round(lows.min(), 2)
            
            # 17. Volume Analysis
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            indicators['Avg_Volume'] = int(avg_volume)
            indicators['Current_Volume'] = int(current_volume)
            indicators['Volume_Ratio'] = round(current_volume / avg_volume, 2)
            
            # 18. Price Change
            current_price = df['Close'].iloc[-1]
            price_10_back = df['Close'].iloc[-10]
            indicators['Price_Change_10'] = round(((current_price - price_10_back) / price_10_back) * 100, 2)
            
            # 19. Trend Direction
            if indicators['SMA_20'] > indicators['SMA_50']:
                indicators['Trend'] = "BULLISH"
            elif indicators['SMA_20'] < indicators['SMA_50']:
                indicators['Trend'] = "BEARISH"
            else:
                indicators['Trend'] = "SIDEWAYS"
            
            return indicators
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return None
    
    @staticmethod
    def generate_signals(indicators, current_price):
        """
        Generate trading signals based on multiple indicators
        Multi-timeframe confluence
        """
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
                signal['reasons'].append("✓ Strong bullish MA alignment (EMA9>EMA21>SMA20)")
            elif current_price < indicators['EMA_9'] < indicators['EMA_21'] < indicators['SMA_20']:
                bearish_score += 15
                signal['reasons'].append("✓ Strong bearish MA alignment")
            
            # ADX (Trend Strength)
            adx = indicators.get('ADX', 0)
            if adx > 25:
                if indicators['Trend'] == "BULLISH":
                    bullish_score += 10
                    signal['reasons'].append(f"✓ Strong trend (ADX: {adx})")
                else:
                    bearish_score += 10
                    signal['reasons'].append(f"✓ Strong trend (ADX: {adx})")
            
            # MACD
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_signal', 0)
            if macd > macd_signal and macd > 0:
                bullish_score += 10
                signal['reasons'].append("✓ MACD bullish crossover")
            elif macd < macd_signal and macd < 0:
                bearish_score += 10
                signal['reasons'].append("✓ MACD bearish crossover")
            
            # SuperTrend
            supertrend = indicators.get('SuperTrend', 0)
            if current_price > supertrend:
                bullish_score += 12
                signal['reasons'].append(f"✓ Above SuperTrend (₹{supertrend})")
            elif current_price < supertrend:
                bearish_score += 12
                signal['reasons'].append(f"✓ Below SuperTrend (₹{supertrend})")
            
            # ==================
            # MOMENTUM ANALYSIS
            # ==================
            
            # RSI
            rsi = indicators.get('RSI', 50)
            if 40 < rsi < 70 and indicators['Trend'] == "BULLISH":
                bullish_score += 8
                signal['reasons'].append(f"✓ RSI in bullish zone ({rsi})")
            elif 30 < rsi < 60 and indicators['Trend'] == "BEARISH":
                bearish_score += 8
                signal['reasons'].append(f"✓ RSI in bearish zone ({rsi})")
            elif rsi > 70:
                signal['reasons'].append(f"⚠️ RSI overbought ({rsi})")
            elif rsi < 30:
                signal['reasons'].append(f"⚠️ RSI oversold ({rsi})")
            
            # Stochastic
            stoch_k = indicators.get('Stoch_K', 0)
            stoch_d = indicators.get('Stoch_D', 0)
            if stoch_k > stoch_d and stoch_k < 80:
                bullish_score += 7
                signal['reasons'].append("✓ Stochastic bullish")
            elif stoch_k < stoch_d and stoch_k > 20:
                bearish_score += 7
                signal['reasons'].append("✓ Stochastic bearish")
            
            # CCI
            cci = indicators.get('CCI', 0)
            if 0 < cci < 100:
                bullish_score += 5
            elif -100 < cci < 0:
                bearish_score += 5
            
            # MFI (Money Flow)
            mfi = indicators.get('MFI', 50)
            if mfi > 50 and mfi < 80:
                bullish_score += 6
                signal['reasons'].append(f"✓ Money flowing in (MFI: {mfi})")
            elif mfi < 50 and mfi > 20:
                bearish_score += 6
                signal['reasons'].append(f"✓ Money flowing out (MFI: {mfi})")
            
            # ==================
            # VOLATILITY & BREAKOUT
            # ==================
            
            # Bollinger Bands
            bb_upper = indicators.get('BB_Upper', 0)
            bb_lower = indicators.get('BB_Lower', 0)
            
            if current_price > bb_upper:
                bullish_score += 8
                signal['reasons'].append("✓ Breakout above Bollinger Upper Band")
            elif current_price < bb_lower:
                bearish_score += 8
                signal['reasons'].append("✓ Breakdown below Bollinger Lower Band")
            
            # Support/Resistance
            resistance = indicators.get('Resistance', 0)
            support = indicators.get('Support', 0)
            
            if current_price > (resistance * 0.99):
                bullish_score += 10
                signal['reasons'].append(f"✓ Near/above resistance (₹{resistance})")
            elif current_price < (support * 1.01):
                bearish_score += 10
                signal['reasons'].append(f"✓ Near/below support (₹{support})")
            
            # ==================
            # VOLUME CONFIRMATION
            # ==================
            
            volume_ratio = indicators.get('Volume_Ratio', 1)
            if volume_ratio > 1.5:
                if bullish_score > bearish_score:
                    bullish_score += 10
                    signal['reasons'].append(f"✓ Volume spike {volume_ratio}x (Bullish confirmation)")
                else:
                    bearish_score += 10
                    signal['reasons'].append(f"✓ Volume spike {volume_ratio}x (Bearish confirmation)")
            
            # ==================
            # SIGNAL GENERATION
            # ==================
            
            total_score = bullish_score + bearish_score
            
            if bullish_score > bearish_score and bullish_score >= 50:
                signal['type'] = "BULLISH"
                signal['strength'] = min(100, bullish_score)
                signal['confidence'] = int((bullish_score / (bullish_score + bearish_score)) * 100)
                
                signal['entry'] = current_price
                signal['target1'] = current_price + (indicators['ATR'] * 1.5)
                signal['target2'] = current_price + (indicators['ATR'] * 3)
                signal['sl'] = max(support, current_price - (indicators['ATR'] * 1))
                
            elif bearish_score > bullish_score and bearish_score >= 50:
                signal['type'] = "BEARISH"
                signal['strength'] = min(100, bearish_score)
                signal['confidence'] = int((bearish_score / (bullish_score + bearish_score)) * 100)
                
                signal['entry'] = current_price
                signal['target1'] = current_price - (indicators['ATR'] * 1.5)
                signal['target2'] = current_price - (indicators['ATR'] * 3)
                signal['sl'] = min(resistance, current_price + (indicators['ATR'] * 1))
                
            else:
                return None  # No clear signal
            
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
# CANDLESTICK PATTERNS (Advanced)
# ========================
class CandlestickPatternDetector:
    """Detect 15+ candlestick patterns"""
    
    @staticmethod
    def detect_all_patterns(df):
        """Detect multiple patterns using TA library"""
        patterns = []
        
        try:
            if len(df) < 5:
                return patterns
            
            # Convert to required format
            ohlc = df[['Open', 'High', 'Low', 'Close']].copy()
            
            # Using finta for pattern detection
            
            # 1. Doji
            if TA.DOJI(ohlc).iloc[-1] == True:
                patterns.append("🔵 DOJI - Indecision")
            
            # 2-5. Manual patterns
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            
            body_last = abs(last['Close'] - last['Open'])
            range_last = last['High'] - last['Low']
            
            # Hammer
            if last['Close'] > last['Open']:
                lower_wick = last['Open'] - last['Low']
                upper_wick = last['High'] - last['Close']
                if range_last > 0 and lower_wick > (body_last * 2) and upper_wick < (body_last * 0.5):
                    patterns.append("🔨 HAMMER - Bullish Reversal")
            
            # Shooting Star
            if last['Close'] < last['Open']:
                upper_wick = last['High'] - last['Open']
                lower_wick = last['Close'] - last['Low']
                if range_last > 0 and upper_wick > (body_last * 2) and lower_wick < (body_last * 0.5):
                    patterns.append("⭐ SHOOTING STAR - Bearish")
            
            # Bullish Engulfing
            if prev['Close'] < prev['Open'] and last['Close'] > last['Open']:
                if last['Open'] <= prev['Close'] and last['Close'] >= prev['Open']:
                    patterns.append("🟢 BULLISH ENGULFING")
            
            # Bearish Engulfing
            if prev['Close'] > prev['Open'] and last['Close'] < last['Open']:
                if last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']:
                    patterns.append("🔴 BEARISH ENGULFING")
            
            # Morning Star (3 candles)
            if len(df) >= 3:
                c1 = df.iloc[-3]
                c2 = df.iloc[-2]
                c3 = df.iloc[-1]
                
                if (c1['Close'] < c1['Open'] and 
                    abs(c2['Close'] - c2['Open']) < body_last * 0.5 and
                    c3['Close'] > c3['Open'] and
                    c3['Close'] > (c1['Open'] + c1['Close']) / 2):
                    patterns.append("🌅 MORNING STAR - Bullish")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return patterns


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
        self.tech_analyzer = AdvancedTechnicalAnalyzer()
        self.pattern_detector = CandlestickPatternDetector()
        
        logger.info("🤖 Advanced Trading Bot initialized")
        logger.info("📊 Using: pandas-ta + finta (130+ indicators)")
    
    async def load_security_ids(self):
        """Load security IDs from Dhan CSV"""
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
                                        logger.info(f"✅ {symbol}: {sec_id}")
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
                                        logger.info(f"✅ {symbol}: {sec_id}")
                                        break
                        except:
                            continue
                
                logger.info(f"✅ {len(self.security_id_map)} securities loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading IDs: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Fetch 350+ candles"""
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
                
                if 'open' in data:
                    candles = []
                    for i in range(len(data['open'])):
                        candles.append({
                            'timestamp': data['start_Time'][i] if i < len(data['start_Time']) else '',
                            'open': data['open'][i],
                            'high': data['high'][i],
                            'low': data['low'][i],
                            'close': data['close'][i],
                            'volume': data['volume'][i]
                        })
                    
                    logger.info(f"{symbol}: {len(candles)} candles")
                    return candles
            
            return None
        except Exception as e:
            logger.error(f"Data error for {symbol}: {e}")
            return None
    
    async def scan_and_alert(self, symbol):
        """Main scanning logic"""
        try:
            if symbol not in self.security_id_map:
                return
            
            info = self.security_id_map[symbol]
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"🔍 {symbol}...")
            
            # Get data
            candles = self.get_historical_data(security_id, segment, symbol)
            if not candles or len(candles) < 50:
                return
            
            # Prepare DataFrame
            df = self.tech_analyzer.prepare_dataframe(candles)
            if df is None:
                return
            
            current_price = df['Close'].iloc[-1]
            
            # Calculate all indicators
            indicators = self.tech_analyzer.calculate_all_indicators(df)
            if not indicators:
                return
            
            # Generate signals
            signal = self.tech_analyzer.generate_signals(indicators, current_price)
            if not signal:
                logger.info(f"⏭️ {symbol}: No signal")
                return
            
            logger.info(f"🎯 {symbol}: {signal['type']} | Confidence: {signal['confidence']}%")
            
            # Detect patterns
            patterns = self.pattern_detector.detect_all_patterns(df)
            
            # Send alert
            await self.send_telegram_alert(symbol, current_price, signal, indicators, patterns)
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
    
    async def send_telegram_alert(self, symbol, price, signal, indicators, patterns):
        """Send formatted alert"""
        try:
            msg = f"🚨 *TRADE SIGNAL* 🚨\n"
            msg += f"{'='*40}\n\n"
            
            msg += f"📊 *{symbol}*\n"
            msg += f"💰 Price: ₹{price:,.2f}\n"
            msg += f"🎯 Signal: *{signal['type']}*\n"
            msg += f"💪 Strength: {signal['strength']}%\n"
            msg += f"✅ Confidence: {signal['confidence']}%\n\n"
            
            msg += f"{'='*40}\n"
            msg += f"📈 *TRADE SETUP*\n"
            msg += f"{'='*40}\n\n"
            
            msg += f"🎯 Entry: ₹{signal['entry']:,.2f}\n"
            
            if signal['target1']:
                gain1 = ((signal['target1'] - signal['entry']) / signal['entry']) * 100
                msg += f"🟢 T1: ₹{signal['target1']:,.2f} ({abs(gain1):.1f}%)\n"
            
            if signal['target2']:
                gain2 = ((signal['target2'] - signal['entry']) / signal['entry']) * 100
                msg += f"🟢 T2: ₹{signal['target2']:,.2f} ({abs(gain2):.1f}%)\n"
            
            if signal['sl']:
                loss = ((signal['entry'] - signal['sl']) / signal['entry']) * 100
                msg += f"🛑 SL: ₹{signal['sl']:,.2f} ({abs(loss):.1f}%)\n"
            
            msg += f"\n📊 R:R = 1:{signal['risk_reward']}\n\n"
            
            # Technical Indicators Summary
            msg += f"{'='*40}\n"
            msg += f"📊 *KEY INDICATORS*\n"
            msg += f"{'='*40}\n\n"
            
            msg += f"🔵 Trend: {indicators.get('Trend', 'N/A')}\n"
            msg += f"📈 SMA20: ₹{indicators.get('SMA_20', 0):,.1f}\n"
            msg += f"📈 EMA9: ₹{indicators.get('EMA_9', 0):,.1f}\n"
            msg += f"📊 RSI: {indicators.get('RSI', 0):.1f}\n"
            msg += f"💹 MACD: {indicators.get('MACD', 0):.2f}\n"
            msg += f"💪 ADX: {indicators.get('ADX', 0):.1f}\n"
            msg += f"💰 MFI: {indicators.get('MFI', 0):.1f}\n"
            msg += f"📊 ATR: ₹{indicators.get('ATR', 0):.2f}\n"
            msg += f"📈 Volume: {indicators.get('Volume_Ratio', 0):.1f}x avg\n\n"
            
            # Reasons
            if signal['reasons']:
                msg += f"{'='*40}\n"
                msg += f"💡 *WHY THIS TRADE?*\n"
                msg += f"{'='*40}\n\n"
                for reason in signal['reasons'][:8]:  # Top 8 reasons
                    msg += f"{reason}\n"
                msg += "\n"
            
            # Patterns
            if patterns:
                msg += f"{'='*40}\n"
                msg += f"🕯️ *PATTERNS DETECTED*\n"
                msg += f"{'='*40}\n\n"
                for pattern in patterns:
                    msg += f"{pattern}\n"
                msg += "\n"
            
            # Support/Resistance
            msg += f"{'='*40}\n"
            msg += f"🎯 *KEY LEVELS*\n"
            msg += f"{'='*40}\n\n"
            msg += f"🔴 Resistance: ₹{indicators.get('Resistance', 0):,.2f}\n"
            msg += f"🟢 Support: ₹{indicators.get('Support', 0):,.2f}\n"
            msg += f"📊 BB Upper: ₹{indicators.get('BB_Upper', 0):,.2f}\n"
            msg += f"📊 BB Lower: ₹{indicators.get('BB_Lower', 0):,.2f}\n"
            msg += f"🎯 VWAP: ₹{indicators.get('VWAP', 0):,.2f}\n\n"
            
            msg += f"⏰ {datetime.now().strftime('%d-%m-%Y %H:%M IST')}\n"
            msg += f"{'='*40}"
            
            # Send message
            if len(msg) > 4000:
                parts = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
                for part in parts:
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=part,
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(1)
            else:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg,
                    parse_mode='Markdown'
                )
            
            logger.info(f"✅ {symbol} alert sent!")
            
        except Exception as e:
            logger.error(f"Alert error: {e}")
    
    async def send_startup_message(self):
        """Startup message"""
        try:
            msg = "🤖 *ADVANCED TRADING BOT ACTIVATED*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            msg += f"📊 *Symbols:* {len(self.security_id_map)}\n"
            msg += f"⏱️ *Interval:* 15 minutes\n"
            msg += f"📈 *Timeframe:* 5-min candles\n\n"
            
            msg += "🎯 *INDICATORS (20+):*\n"
            msg += "  ✓ SMA, EMA, VWAP\n"
            msg += "  ✓ RSI, Stochastic, Williams %R\n"
            msg += "  ✓ MACD, ADX, SuperTrend\n"
            msg += "  ✓ Bollinger Bands, Keltner\n"
            msg += "  ✓ ATR, CCI, MFI\n"
            msg += "  ✓ OBV, A/D Line\n"
            msg += "  ✓ Volume analysis\n\n"
            
            msg += "📊 *PATTERNS:*\n"
            msg += "  ✓ Doji, Hammer, Shooting Star\n"
            msg += "  ✓ Engulfing patterns\n"
            msg += "  ✓ Morning/Evening Star\n\n"
            
            msg += "🔥 *POWERED BY:*\n"
            msg += "  • pandas-ta (130+ indicators)\n"
            msg += "  • finta (80+ indicators)\n"
            msg += "  • DhanHQ API v2\n\n"
            
            msg += "💡 *FILTERS:*\n"
            msg += "  • Min Confidence: 70%\n"
            msg += "  • Min R:R: 1.5\n"
            msg += "  • Multi-indicator confluence\n\n"
            
            msg += "🔔 *Status:* ACTIVE ✅\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            
            logger.info("✅ Startup message sent")
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("🚀 Starting Advanced Bot...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load IDs")
            return
        
        await self.send_startup_message()
        
        symbols = list(self.security_id_map.keys())
        
        while self.running:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 SCAN: {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"{'='*60}\n")
                
                for idx, symbol in enumerate(symbols, 1):
                    logger.info(f"[{idx}/{len(symbols)}] {symbol}")
                    await self.scan_and_alert(symbol)
                    
                    if idx < len(symbols):
                        await asyncio.sleep(8)
                
                logger.info("\n✅ Cycle complete! Next in 15 min...\n")
                await asyncio.sleep(900)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error: {e}")
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
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN
        }
        
        missing = [k for k, v in required.items() if not v]
        
        if missing:
            logger.error(f"❌ Missing: {', '.join(missing)}")
            exit(1)
        
        logger.info("✅ Environment OK")
        logger.info("🚀 Launching bot...")
        
        bot = TradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 FATAL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
