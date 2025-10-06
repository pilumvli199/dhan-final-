import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import csv
import json
from openai import OpenAI

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Railway-specific: Keep process alive
PORT = os.getenv("PORT", "8080")

# OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# Stock/Index List - Extended
STOCKS_INDICES = {
    # Indices
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
    "FINNIFTY": {"symbol": "FINNIFTY", "segment": "IDX_I"},
    
    # High Volume Stocks
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ"},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
    "LTIM": {"symbol": "LTIM", "segment": "NSE_EQ"},
    "ADANIENT": {"symbol": "ADANIENT", "segment": "NSE_EQ"},
    "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ"},
    "LT": {"symbol": "LT", "segment": "NSE_EQ"},
    "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ"},
    "TECHM": {"symbol": "TECHM", "segment": "NSE_EQ"},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ"},
    "DRREDDY": {"symbol": "DRREDDY", "segment": "NSE_EQ"},
    "WIPRO": {"symbol": "WIPRO", "segment": "NSE_EQ"},
    "TRENT": {"symbol": "TRENT", "segment": "NSE_EQ"},
    "TITAN": {"symbol": "TITAN", "segment": "NSE_EQ"},
    "ASIANPAINT": {"symbol": "ASIANPAINT", "segment": "NSE_EQ"},
}

# ========================
# AI OPTION BOT
# ========================

class AIOptionTradingBot:
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
        logger.info("🤖 AI Option Trading Bot initialized")
    
    async def load_security_ids(self):
        """Load security IDs from Dhan"""
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
                            # Index
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
                            # Stock
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
    
    def get_historical_data(self, security_id, segment, candles=100):
        """Get 5-min historical candles (100 candles = ~8.3 hours)"""
        try:
            # Calculate from_date (100 candles @ 5min = ~500 minutes = ~8.3 hours)
            # Add buffer for non-trading hours
            from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            payload = {
                "securityId": security_id,
                "exchangeSegment": segment,
                "instrument": "EQUITY" if segment == "NSE_EQ" else "INDEX",
                "expiryCode": 0,
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = requests.post(
                DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    # Get last 100 5-min candles
                    candles_data = data['data']
                    if isinstance(candles_data, list) and len(candles_data) > 0:
                        # Take last 100 candles
                        return candles_data[-candles:]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """Get nearest expiry"""
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
                        return expiries[0]  # Nearest expiry
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting expiry: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Get option chain data"""
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
    
    def calculate_technical_indicators(self, candles):
        """Calculate key technical indicators"""
        try:
            closes = [float(c['close']) for c in candles]
            highs = [float(c['high']) for c in candles]
            lows = [float(c['low']) for c in candles]
            volumes = [float(c['volume']) for c in candles]
            
            # Support/Resistance (last 20 candles)
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            resistance = max(recent_highs)
            support = min(recent_lows)
            
            # ATR (14 periods)
            tr_list = []
            for i in range(1, min(15, len(candles))):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])
                tr = max(high_low, high_close, low_close)
                tr_list.append(tr)
            
            atr = sum(tr_list) / len(tr_list) if tr_list else 0
            
            # Price change %
            price_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            # Volume analysis
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_spike = (current_volume / avg_volume) if avg_volume > 0 else 1
            
            return {
                "current_price": closes[-1],
                "support": support,
                "resistance": resistance,
                "atr": atr,
                "price_change_pct": price_change_pct,
                "volume_spike": volume_spike,
                "avg_volume": avg_volume
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def analyze_option_chain(self, oc_data, spot_price):
        """Analyze option chain for key metrics"""
        try:
            oc = oc_data.get('oc', {})
            if not oc:
                return None
            
            total_ce_oi = 0
            total_pe_oi = 0
            total_ce_volume = 0
            total_pe_volume = 0
            
            # Find ATM strike
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            # Get ATM ± 3 strikes data
            atm_idx = strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 3)
            end_idx = min(len(strikes), atm_idx + 4)
            relevant_strikes = strikes[start_idx:end_idx]
            
            max_ce_oi_strike = None
            max_pe_oi_strike = None
            max_ce_oi = 0
            max_pe_oi = 0
            
            for strike in relevant_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                ce_oi = ce.get('oi', 0)
                pe_oi = pe.get('oi', 0)
                ce_vol = ce.get('volume', 0)
                pe_vol = pe.get('volume', 0)
                
                total_ce_oi += ce_oi
                total_pe_oi += pe_oi
                total_ce_volume += ce_vol
                total_pe_volume += pe_vol
                
                if ce_oi > max_ce_oi:
                    max_ce_oi = ce_oi
                    max_ce_oi_strike = strike
                
                if pe_oi > max_pe_oi:
                    max_pe_oi = pe_oi
                    max_pe_oi_strike = strike
            
            # PCR (Put-Call Ratio)
            pcr = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 0
            
            # ATM data
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            atm_ce = atm_data.get('ce', {})
            atm_pe = atm_data.get('pe', {})
            
            return {
                "pcr": pcr,
                "atm_strike": atm_strike,
                "max_pain": max_pe_oi_strike,  # Simplified max pain
                "ce_total_oi": total_ce_oi,
                "pe_total_oi": total_pe_oi,
                "ce_total_volume": total_ce_volume,
                "pe_total_volume": total_pe_volume,
                "atm_ce_price": atm_ce.get('last_price', 0),
                "atm_pe_price": atm_pe.get('last_price', 0),
                "atm_ce_iv": atm_ce.get('implied_volatility', 0),
                "atm_pe_iv": atm_pe.get('implied_volatility', 0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option chain: {e}")
            return None
    
    async def get_ai_analysis(self, symbol, candles, technical_data, option_data, spot_price):
        """Get AI analysis from GPT-4o-mini"""
        try:
            # Prepare candles summary (last 10 candles)
            recent_candles = candles[-10:]
            candles_summary = []
            for c in recent_candles:
                candles_summary.append({
                    "time": c.get('timestamp', 'N/A'),
                    "open": c.get('open'),
                    "high": c.get('high'),
                    "low": c.get('low'),
                    "close": c.get('close'),
                    "volume": c.get('volume')
                })
            
            # Create AI prompt
            prompt = f"""You are an expert option trader analyzing {symbol}. Provide a trading signal based on the following data:

**Current Spot Price:** ₹{spot_price:,.2f}

**Technical Analysis (100 candles, 5-min timeframe):**
- Support: ₹{technical_data['support']:,.2f}
- Resistance: ₹{technical_data['resistance']:,.2f}
- ATR: ₹{technical_data['atr']:.2f}
- Price Change: {technical_data['price_change_pct']:.2f}%
- Volume Spike: {technical_data['volume_spike']:.2f}x

**Recent 10 Candles (5-min):**
{json.dumps(candles_summary, indent=2)}

**Option Chain Data:**
- PCR Ratio: {option_data['pcr']:.2f}
- ATM Strike: ₹{option_data['atm_strike']:,.0f}
- Max Pain: ₹{option_data['max_pain']:,.0f}
- CE Total OI: {option_data['ce_total_oi']:,}
- PE Total OI: {option_data['pe_total_oi']:,}
- ATM CE Price: ₹{option_data['atm_ce_price']:.2f}, IV: {option_data['atm_ce_iv']:.1f}%
- ATM PE Price: ₹{option_data['atm_pe_price']:.2f}, IV: {option_data['atm_pe_iv']:.1f}%

**Trading Strategy Rules:**
1. **Breakout**: Price breaks resistance + Call OI buildup → BUY CE
2. **Reversal**: Price bounces from support + Put OI increase → BUY PE
3. **Momentum**: Strong candle + volume spike + favorable PCR → BUY CE/PE
4. **Mean Reversion**: Oversold/Overbought + contrarian OI → BUY opposite

**Your Task:**
Analyze the data and provide a trading signal in this EXACT JSON format:
{{
    "signal": "BUY_CE" or "BUY_PE" or "NO_TRADE",
    "confidence": 0-100,
    "entry_price": price,
    "stop_loss": price,
    "target": price,
    "strike": strike_price,
    "reasoning": "brief explanation (2-3 lines)",
    "risk_reward": ratio
}}

**Important:**
- Only give signal if confidence ≥ 60%
- Ensure 1:2 minimum risk-reward ratio
- Consider both price action AND option chain
- Be conservative, quality > quantity
"""

            # Call GPT-4o-mini
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert option trader. Respond ONLY with valid JSON, no extra text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            ai_response = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()
            
            signal_data = json.loads(ai_response)
            
            logger.info(f"AI Signal for {symbol}: {signal_data.get('signal')} (Confidence: {signal_data.get('confidence')}%)")
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return None
    
    def format_signal_message(self, symbol, signal_data, spot_price, expiry):
        """Format trading signal for Telegram"""
        try:
            signal_type = signal_data.get('signal')
            
            if signal_type == "NO_TRADE":
                return None  # Don't send message for no trade
            
            confidence = signal_data.get('confidence', 0)
            
            # Signal emoji
            signal_emoji = "🟢 BUY CALL" if signal_type == "BUY_CE" else "🔴 BUY PUT"
            
            msg = f"{signal_emoji}\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"📊 *{symbol}*\n"
            msg += f"💰 Spot: ₹{spot_price:,.2f}\n"
            msg += f"📅 Expiry: {expiry}\n\n"
            
            msg += f"🎯 *Trade Setup:*\n"
            msg += f"Strike: ₹{signal_data.get('strike', 0):,.0f}\n"
            msg += f"Entry: ₹{signal_data.get('entry_price', 0):.2f}\n"
            msg += f"SL: ₹{signal_data.get('stop_loss', 0):.2f}\n"
            msg += f"Target: ₹{signal_data.get('target', 0):.2f}\n"
            msg += f"R:R = 1:{signal_data.get('risk_reward', 0):.1f}\n\n"
            
            msg += f"📈 *Confidence:* {confidence}%\n"
            msg += f"{'🟩' * int(confidence/10)}{'⬜' * (10-int(confidence/10))}\n\n"
            
            msg += f"💡 *Reasoning:*\n_{signal_data.get('reasoning', 'N/A')}_\n\n"
            
            msg += f"⚠️ *Risk Management:*\n"
            msg += f"• Max risk: 2-3% of capital\n"
            msg += f"• Exit at SL, no averaging\n"
            msg += f"• Book 50% at 1:1, trail rest\n\n"
            
            msg += f"⏰ {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
            
            return msg
            
        except Exception as e:
            logger.error(f"Error formatting signal message: {e}")
            return None
    
    async def analyze_and_send_signals(self, symbols_batch):
        """Analyze batch and send signals"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                
                logger.info(f"\n{'='*50}")
                logger.info(f"Analyzing {symbol}...")
                logger.info(f"{'='*50}")
                
                # 1. Get historical candles
                candles = self.get_historical_data(security_id, segment)
                if not candles or len(candles) < 50:
                    logger.warning(f"{symbol}: Not enough candle data")
                    continue
                
                logger.info(f"✅ Got {len(candles)} candles")
                
                # 2. Calculate technical indicators
                technical_data = self.calculate_technical_indicators(candles)
                if not technical_data:
                    continue
                
                spot_price = technical_data['current_price']
                logger.info(f"✅ Technical analysis done. Spot: ₹{spot_price:,.2f}")
                
                # 3. Get option chain
                expiry = self.get_nearest_expiry(security_id, segment)
                if not expiry:
                    logger.warning(f"{symbol}: No expiry found")
                    continue
                
                oc_data = self.get_option_chain(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"{symbol}: No option chain data")
                    continue
                
                logger.info(f"✅ Option chain fetched (Expiry: {expiry})")
                
                # 4. Analyze option chain
                option_analysis = self.analyze_option_chain(oc_data, spot_price)
                if not option_analysis:
                    continue
                
                logger.info(f"✅ Option analysis done. PCR: {option_analysis['pcr']:.2f}")
                
                # 5. Get AI signal
                signal_data = await self.get_ai_analysis(
                    symbol, candles, technical_data, option_analysis, spot_price
                )
                
                if not signal_data:
                    logger.warning(f"{symbol}: AI analysis failed")
                    continue
                
                # 6. Send signal if actionable
                if signal_data.get('signal') != 'NO_TRADE' and signal_data.get('confidence', 0) >= 60:
                    message = self.format_signal_message(symbol, signal_data, spot_price, expiry)
                    if message:
                        await self.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=message,
                            parse_mode='Markdown'
                        )
                        logger.info(f"🚀 Signal sent for {symbol}!")
                else:
                    logger.info(f"⏭️ {symbol}: No actionable signal (Confidence: {signal_data.get('confidence', 0)}%)")
                
                # Rate limit
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                await asyncio.sleep(3)
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 AI Option Trading Bot starting...")
        
        # Load security IDs
        success = await self.load_security_ids()
        if not success:
            logger.error("Failed to load security IDs. Exiting...")
            return
        
        await self.send_startup_message()
        
        # Divide into batches
        all_symbols = list(self.security_id_map.keys())
        batch_size = 3  # Smaller batches due to AI processing
        batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
        
        logger.info(f"Total {len(all_symbols)} symbols in {len(batches)} batches")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Starting analysis cycle at {timestamp}")
                logger.info(f"{'='*60}")
                
                for batch_num, batch in enumerate(batches, 1):
                    logger.info(f"\n📦 Batch {batch_num}/{len(batches)}: {batch}")
                    await self.analyze_and_send_
