import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import csv
import pandas as pd
import numpy as np
import json

# Technical Analysis
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
# TECHNICAL ANALYZER
# ========================
class TechnicalAnalyzer:
    
    @staticmethod
    def prepare_dataframe(candles):
        try:
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            logger.error(f"DataFrame error: {e}")
            return None
    
    @staticmethod
    def calculate_indicators(df):
        try:
            if df is None or len(df) < 50:
                return None
            
            indicators = {}
            df = df.copy()
            
            # Moving Averages
            indicators['SMA_20'] = round(ta.trend.sma_indicator(df['close'], window=20).iloc[-1], 2)
            indicators['SMA_50'] = round(ta.trend.sma_indicator(df['close'], window=50).iloc[-1], 2)
            indicators['EMA_9'] = round(ta.trend.ema_indicator(df['close'], window=9).iloc[-1], 2)
            indicators['EMA_21'] = round(ta.trend.ema_indicator(df['close'], window=21).iloc[-1], 2)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            indicators['MACD'] = round(macd.macd().iloc[-1], 2)
            indicators['MACD_signal'] = round(macd.macd_signal().iloc[-1], 2)
            
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            indicators['ADX'] = round(adx.adx().iloc[-1], 2)
            
            # RSI
            indicators['RSI'] = round(ta.momentum.rsi(df['close'], window=14).iloc[-1], 2)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            indicators['Stoch_K'] = round(stoch.stoch().iloc[-1], 2)
            indicators['Stoch_D'] = round(stoch.stoch_signal().iloc[-1], 2)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            indicators['BB_Upper'] = round(bb.bollinger_hband().iloc[-1], 2)
            indicators['BB_Lower'] = round(bb.bollinger_lband().iloc[-1], 2)
            
            # ATR
            indicators['ATR'] = round(ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1], 2)
            
            # MFI
            indicators['MFI'] = round(ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume']).iloc[-1], 2)
            
            # Support/Resistance
            indicators['Resistance'] = round(df['high'].tail(50).max(), 2)
            indicators['Support'] = round(df['low'].tail(50).min(), 2)
            
            # Volume
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            indicators['Volume_Ratio'] = round(current_volume / avg_volume, 2) if avg_volume > 0 else 0
            
            # Trend
            if indicators['SMA_20'] > indicators['SMA_50']:
                indicators['Trend'] = "BULLISH"
            elif indicators['SMA_20'] < indicators['SMA_50']:
                indicators['Trend'] = "BEARISH"
            else:
                indicators['Trend'] = "SIDEWAYS"
            
            return indicators
            
        except Exception as e:
            logger.error(f"Indicator error: {e}")
            return None
    
    @staticmethod
    def generate_signals(indicators, current_price):
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
            
            # MA Alignment
            if current_price > indicators['EMA_9'] > indicators['EMA_21']:
                bullish_score += 15
                signal['reasons'].append("✓ Bullish MA alignment")
            elif current_price < indicators['EMA_9'] < indicators['EMA_21']:
                bearish_score += 15
                signal['reasons'].append("✓ Bearish MA alignment")
            
            # ADX
            adx = indicators.get('ADX', 0)
            if adx > 25:
                if indicators['Trend'] == "BULLISH":
                    bullish_score += 10
                    signal['reasons'].append(f"✓ Strong trend (ADX: {adx})")
                else:
                    bearish_score += 10
            
            # MACD
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_signal', 0)
            if macd > macd_signal:
                bullish_score += 10
                signal['reasons'].append("✓ MACD bullish")
            elif macd < macd_signal:
                bearish_score += 10
            
            # RSI
            rsi = indicators.get('RSI', 50)
            if 40 < rsi < 70:
                bullish_score += 8
                signal['reasons'].append(f"✓ RSI: {rsi:.1f}")
            elif 30 < rsi < 60:
                bearish_score += 8
            
            # Stochastic
            stoch_k = indicators.get('Stoch_K', 50)
            stoch_d = indicators.get('Stoch_D', 50)
            if stoch_k > stoch_d and stoch_k < 80:
                bullish_score += 7
            elif stoch_k < stoch_d and stoch_k > 20:
                bearish_score += 7
            
            # MFI
            mfi = indicators.get('MFI', 50)
            if mfi > 50 and mfi < 80:
                bullish_score += 6
                signal['reasons'].append(f"✓ MFI: {mfi:.1f}")
            elif mfi < 50:
                bearish_score += 6
            
            # Bollinger
            bb_upper = indicators.get('BB_Upper', 0)
            bb_lower = indicators.get('BB_Lower', 0)
            if current_price > bb_upper:
                bullish_score += 8
                signal['reasons'].append("✓ Above BB upper")
            elif current_price < bb_lower:
                bearish_score += 8
            
            # Volume
            volume_ratio = indicators.get('Volume_Ratio', 1)
            if volume_ratio > 1.5:
                if bullish_score > bearish_score:
                    bullish_score += 10
                    signal['reasons'].append(f"✓ Volume: {volume_ratio:.1f}x")
                else:
                    bearish_score += 10
            
            # Generate Signal
            atr = indicators.get('ATR', 0)
            support = indicators.get('Support', 0)
            resistance = indicators.get('Resistance', 0)
            
            if bullish_score > bearish_score and bullish_score >= 50:
                signal['type'] = "BULLISH"
                signal['strength'] = min(100, bullish_score)
                signal['confidence'] = int((bullish_score / (bullish_score + bearish_score)) * 100)
                signal['entry'] = current_price
                signal['target1'] = current_price + (atr * 1.5) if atr > 0 else current_price * 1.02
                signal['target2'] = current_price + (atr * 3) if atr > 0 else current_price * 1.04
                signal['sl'] = max(support, current_price - atr) if atr > 0 else support
                
            elif bearish_score > bullish_score and bearish_score >= 50:
                signal['type'] = "BEARISH"
                signal['strength'] = min(100, bearish_score)
                signal['confidence'] = int((bearish_score / (bullish_score + bearish_score)) * 100)
                signal['entry'] = current_price
                signal['target1'] = current_price - (atr * 1.5) if atr > 0 else current_price * 0.98
                signal['target2'] = current_price - (atr * 3) if atr > 0 else current_price * 0.96
                signal['sl'] = min(resistance, current_price + atr) if atr > 0 else resistance
            else:
                return None
            
            # R:R
            if signal['sl'] and signal['target1']:
                risk = abs(signal['entry'] - signal['sl'])
                reward = abs(signal['target1'] - signal['entry'])
                signal['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
            
            if signal['confidence'] < 70 or signal['risk_reward'] < 1.5:
                return None
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal error: {e}")
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
        self.analyzer = TechnicalAnalyzer()
        logger.info("🤖 Bot initialized")
    
    async def load_security_ids(self):
        try:
            logger.info("Loading IDs...")
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
                
                # Check what keys are in response
                logger.info(f"API Response keys for {symbol}: {list(data.keys())}")
                
                if 'open' in data:
                    candles = []
                    
                    # Try different possible timestamp keys
                    timestamps = []
                    if 'start_Time' in data:
                        timestamps = data['start_Time']
                    elif 'timestamp' in data:
                        timestamps = data['timestamp']
                    elif 'time' in data:
                        timestamps = data['time']
                    else:
                        # Generate timestamps if not provided
                        timestamps = [f"2025-10-16 09:{15+i}:00" for i in range(len(data['open']))]
                    
                    for i in range(len(data['open'])):
                        candles.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else '',
                            'open': data['open'][i],
                            'high': data['high'][i],
                            'low': data['low'][i],
                            'close': data['close'][i],
                            'volume': data['volume'][i]
                        })
                    
                    logger.info(f"{symbol}: Fetched {len(candles)} candles")
                    return candles
                else:
                    logger.error(f"{symbol}: No OHLC data in response")
            else:
                logger.error(f"{symbol}: API returned status {response.status_code}")
            
            return None
        except Exception as e:
            logger.error(f"Data error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def scan_and_alert(self, symbol):
        try:
            if symbol not in self.security_id_map:
                return
            
            info = self.security_id_map[symbol]
            
            candles = self.get_historical_data(info['security_id'], info['segment'], symbol)
            if not candles or len(candles) < 50:
                return
            
            df = self.analyzer.prepare_dataframe(candles)
            if df is None:
                return
            
            current_price = df['close'].iloc[-1]
            
            indicators = self.analyzer.calculate_indicators(df)
            if not indicators:
                return
            
            signal = self.analyzer.generate_signals(indicators, current_price)
            if not signal:
                logger.info(f"⏭️ {symbol}: No signal")
                return
            
            logger.info(f"🎯 {symbol}: {signal['type']} ({signal['confidence']}%)")
            await self.send_alert(symbol, current_price, signal, indicators)
            
        except Exception as e:
            logger.error(f"Scan error {symbol}: {e}")
    
    async def send_alert(self, symbol, price, signal, indicators):
        try:
            msg = f"🚨 *TRADE SIGNAL*\n{'='*35}\n\n"
            msg += f"📊 *{symbol}* | ₹{price:,.2f}\n"
            msg += f"🎯 *{signal['type']}*\n"
            msg += f"💪 Strength: {signal['strength']}%\n"
            msg += f"✅ Confidence: {signal['confidence']}%\n\n"
            
            msg += f"{'='*35}\n📈 *SETUP*\n{'='*35}\n\n"
            msg += f"🎯 Entry: ₹{signal['entry']:,.2f}\n"
            
            if signal['target1']:
                gain = abs((signal['target1']-signal['entry'])/signal['entry']*100)
                msg += f"🟢 T1: ₹{signal['target1']:,.2f} ({gain:.1f}%)\n"
            
            if signal['target2']:
                gain = abs((signal['target2']-signal['entry'])/signal['entry']*100)
                msg += f"🟢 T2: ₹{signal['target2']:,.2f} ({gain:.1f}%)\n"
            
            if signal['sl']:
                loss = abs((signal['entry']-signal['sl'])/signal['entry']*100)
                msg += f"🛑 SL: ₹{signal['sl']:,.2f} ({loss:.1f}%)\n"
            
            msg += f"\n📊 R:R = 1:{signal['risk_reward']}\n\n"
            
            msg += f"{'='*35}\n📊 *INDICATORS*\n{'='*35}\n\n"
            msg += f"Trend: {indicators['Trend']}\n"
            msg += f"RSI: {indicators['RSI']:.1f}\n"
            msg += f"ADX: {indicators['ADX']:.1f}\n"
            msg += f"MFI: {indicators['MFI']:.1f}\n"
            msg += f"Volume: {indicators['Volume_Ratio']:.1f}x\n\n"
            
            if signal['reasons']:
                msg += f"{'='*35}\n💡 *REASONS*\n{'='*35}\n\n"
                for reason in signal['reasons'][:6]:
                    msg += f"{reason}\n"
                msg += "\n"
            
            msg += f"⏰ {datetime.now().strftime('%H:%M IST')}\n"
            msg += f"{'='*35}"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Alert sent: {symbol}")
            
        except Exception as e:
            logger.error(f"Alert error: {e}")
    
    async def send_startup_message(self):
        try:
            msg = "🤖 *BOT ACTIVE*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━\n\n"
            msg += f"📊 Symbols: {len(self.security_id_map)}\n"
            msg += f"⏱️ Interval: 15 min\n"
            msg += f"📈 Timeframe: 5-min\n\n"
            msg += "🎯 *INDICATORS:*\n"
            msg += "  ✓ SMA, EMA, MACD, ADX\n"
            msg += "  ✓ RSI, Stochastic, MFI\n"
            msg += "  ✓ Bollinger Bands, ATR\n"
            msg += "  ✓ Volume analysis\n\n"
            msg += "🔔 Status: ACTIVE ✅\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup sent")
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    async def run(self):
        logger.info("🚀 Starting bot...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load IDs")
            return
        
        await self.send_startup_message()
        
        symbols = list(self.security_id_map.keys())
        
        while self.running:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🔄 SCAN: {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"{'='*50}\n")
                
                for idx, symbol in enumerate(symbols, 1):
                    logger.info(f"[{idx}/{len(symbols)}] {symbol}")
                    await self.scan_and_alert(symbol)
                    
                    if idx < len(symbols):
                        await asyncio.sleep(8)
                
                logger.info("\n✅ Cycle done! Next in 15 min...\n")
                await asyncio.sleep(900)
                
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
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN
        }
        
        missing = [k for k, v in required.items() if not v]
        
        if missing:
            logger.error(f"❌ Missing: {', '.join(missing)}")
            exit(1)
        
        logger.info("✅ Environment OK")
        logger.info("🚀 Starting...")
        
        bot = TradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 FATAL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
