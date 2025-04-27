# Core Python
import os
import io
import sys
import time
import logging
from datetime import datetime, timedelta

# Data Handling
import numpy as np
import pandas as pd
import yaml

# Charting
import matplotlib.pyplot as plt

# MT5
import MetaTrader5 as mt5

# Telegram (v20.8 syntax)
from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters as fts
)

# Constants
CONFIG_FILE = "config.yaml"
LOG_FILE = "vwap_bot.log"
PID_FILE = "vwap_bot.pid"

class VWAPMasterBot:
    def __init__(self):
        self.cfg = self._load_config()
        self.application = Application.builder().token(self.cfg['telegram']['token']).build()
        self.logger = self._setup_logging()
        self.running = False
        self.mt5_connected = False
        
        # Register handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("stop", self.stop))
        self.application.add_error_handler(self._error_handler)

    def _load_config(self):
        """Load and validate configuration"""
        with open(CONFIG_FILE) as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required = {
            'telegram': ['token', 'chat_id'],
            'mt5': ['account', 'password', 'server'],
            'symbols': None,
            'timeframes': None
        }
        
        for section, fields in required.items():
            if section not in config:
                raise ValueError(f"Missing config section: {section}")
            if fields and any(field not in config[section] for field in fields):
                raise ValueError(f"Missing required fields in {section}")
        
        # Validate timeframes
        valid_tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
        for tf in config['timeframes']:
            if tf not in valid_tfs:
                raise ValueError(f"Invalid timeframe: {tf}")
        
        return config

    def _setup_logging(self):
        """Configure professional logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('VWAPMaster')

    async def _connect_mt5(self):
        """Establish MT5 connection with retries"""
        try:
            if not mt5.initialize(
                path=self.cfg['mt5'].get('path'),  # path is optional
                login=int(self.cfg['mt5']['account']),
                password=str(self.cfg['mt5']['password']),
                server=str(self.cfg['mt5']['server'])
            ):
                error = mt5.last_error()
                self.logger.error(f"MT5 connection failed: {error}")
                return False
            
            self.mt5_connected = True
            self.logger.info("MT5 connected successfully")
            return True
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False

    def _get_data(self, symbol, timeframe, bars=500):
        """Fetch OHLCV data for specified timeframe"""
        tf_mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4
        }
        
        rates = mt5.copy_rates_from_pos(
            symbol, 
            tf_mapping[timeframe], 
            0, 
            bars
        )
        
        if rates is None:
            self.logger.error(f"No data for {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def _calculate_indicators(self, df, timeframe):
        """Calculate all technical indicators"""
        # Price Source
        price_src = {
            'typical': (df['high'] + df['low'] + df['close']) / 3,
            'close': df['close'],
            'high': df['high'],
            'low': df['low']
        }[self.cfg['vwap']['source']]

        # VWAP Calculation
        if timeframe in ["H1", "H4"]:  # Higher timeframes use daily VWAP
            reset_hour = self.cfg['vwap']['daily_reset_hour']
            df['new_day'] = df['time'].dt.hour == reset_hour
            df['session_id'] = df['new_day'].cumsum()
            
            # Calculate cumulative volume per session
            df['cum_vol'] = df.groupby('session_id', group_keys=False)['real_volume'].cumsum()
            
            # Calculate cumulative price*volume per session
            pv = price_src * df['real_volume']
            df['cum_pv'] = df.groupby('session_id', group_keys=False).apply(
                lambda x: pv[x.index].cumsum()
            ).values
            
            df['vwap'] = df['cum_pv'] / df['cum_vol']
        else:
            # Lower timeframes use continuous VWAP
            df['cum_vol'] = df['real_volume'].cumsum()
            df['cum_pv'] = (price_src * df['real_volume']).cumsum()
            df['vwap'] = df['cum_pv'] / df['cum_vol']

        # AVWAP (Swing-Based)
        swing_idx = df['high'].rolling(self.cfg['avwap']['swing_window']).max().idxmax()
        df_swing = df.iloc[swing_idx:].copy()
        df['avwap'] = (price_src.iloc[swing_idx:] * df_swing['real_volume']).cumsum() / df_swing['real_volume'].cumsum()

        # EMAs
        for period in self.cfg['ma']['ema_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # MACD
        df['macd_line'] = df['close'].ewm(span=self.cfg['macd']['fast']).mean() - \
                         df['close'].ewm(span=self.cfg['macd']['slow']).mean()
        df['signal_line'] = df['macd_line'].ewm(span=self.cfg['macd']['signal']).mean()

        # ATR
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ])
        df['atr'] = df['tr'].rolling(self.cfg['atr']['period']).mean()

        # Volume Analysis
        df['vol_ma'] = df['real_volume'].rolling(20).mean()
        
        return df

    async def _analyze_timeframe(self, symbol, timeframe, chat_id):
        """Core analysis for one symbol/timeframe"""
        df = self._get_data(symbol, timeframe)
        if df is None:
            return

        df = self._calculate_indicators(df, timeframe)
        last = df.iloc[-1]

        # Signal Conditions
        long_cond = (
            (last['close'] > last['vwap']) &
            (last['close'] > last['avwap']) &
            (last['close'] > last['ema_50']) &
            (last['macd_line'] > last['signal_line']) &
            (last['real_volume'] > self.cfg['alert']['volume_spike_multiplier'] * last['vol_ma'])
        )

        short_cond = (
            (last['close'] < last['vwap']) &
            (last['close'] < last['avwap']) &
            (last['close'] < last['ema_50']) &
            (last['macd_line'] < last['signal_line']) &
            (last['real_volume'] > self.cfg['alert']['volume_spike_multiplier'] * last['vol_ma'])
        )

        if long_cond or short_cond:
            signal = "BUY" if long_cond else "SELL"
            sl = last['close'] - (self.cfg['alert']['stop_atr_multiplier'] * last['atr']) if long_cond else \
                 last['close'] + (self.cfg['alert']['stop_atr_multiplier'] * last['atr'])
            
            tp = last['close'] + (self.cfg['alert']['take_profit_atr_multiplier'] * last['atr']) if long_cond else \
                 last['close'] - (self.cfg['alert']['take_profit_atr_multiplier'] * last['atr'])

            chart = self._generate_chart(df, symbol, timeframe)
            await self._send_alert(
                chat_id,
                symbol,
                timeframe,
                signal,
                last['close'],
                last['vwap'],
                last['avwap'],
                last['atr'],
                sl,
                tp,
                chart
            )

    def _generate_chart(self, df, symbol, timeframe):
        """Create professional multi-pane chart"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])

        # Price Plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['close'], label='Price', color='#1f77b4', linewidth=2)
        ax1.plot(df['vwap'], label='VWAP', color='#ff7f0e', linestyle='--')
        ax1.plot(df['avwap'], label='AVWAP', color='#2ca02c', linestyle='-.')
        for period in self.cfg['ma']['ema_periods']:
            ax1.plot(df[f'ema_{period}'], label=f'EMA {period}', alpha=0.7)
        ax1.set_title(f"{symbol} {timeframe} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        ax1.legend(loc='upper left')

        # MACD Plot
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(df.index, df['macd_line'] - df['signal_line'], 
               color=np.where(df['macd_line'] > df['signal_line'], '#2ca02c', '#d62728'),
               alpha=0.6)
        ax2.plot(df['macd_line'], label='MACD', color='#1f77b4')
        ax2.plot(df['signal_line'], label='Signal', color='#ff7f0e')
        ax2.legend(loc='upper left')

        # Volume Plot
        ax3 = fig.add_subplot(gs[2])
        ax3.bar(df.index, df['real_volume'], color='#17becf', alpha=0.6)
        ax3.plot(df['vol_ma'], label='Volume MA', color='#7f7f7f')
        ax3.legend(loc='upper left')

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

    async def _send_alert(self, chat_id, symbol, timeframe, signal, price, vwap, avwap, atr, sl, tp, chart):
        """Send professional trading alert"""
        message = (
            f"🚨 *{symbol} {timeframe} {signal} SIGNAL*\n\n"
            f"• Price: `{price:.5f}`\n"
            f"• VWAP: `{vwap:.5f}`\n"
            f"• AVWAP: `{avwap:.5f}`\n"
            f"• ATR(14): `{atr:.5f}`\n"
            f"• Stop Loss: `{sl:.5f}`\n"
            f"• Take Profit: `{tp:.5f}`\n\n"
            f"Risk: {self.cfg['alert']['risk_percent']}% of balance"
        )
        
        try:
            await self.application.bot.send_photo(
                chat_id=chat_id,
                photo=InputFile(chart, filename=f"{symbol}_{timeframe}_{signal}.png"),
                caption=message,
                parse_mode="Markdown"
            )
            self.logger.info(f"Alert sent for {symbol} {timeframe}")
        except Exception as e:
            self.logger.error(f"Failed to send alert: {str(e)}")

    async def _market_analysis(self, context: ContextTypes.DEFAULT_TYPE):
        """Main analysis loop"""
        if not self.running or not self.mt5_connected:
            return

        chat_id = context.job.chat_id
        for symbol in self.cfg['symbols']:
            for timeframe in self.cfg['timeframes']:
                try:
                    await self._analyze_timeframe(symbol, timeframe, chat_id)
                except Exception as e:
                    self.logger.error(f"Analysis failed for {symbol} {timeframe}: {str(e)}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if self.running:
            await update.message.reply_text("⚠️ Bot is already running!")
            return

        if not await self._connect_mt5():
            await update.message.reply_text("❌ Failed to connect to MT5!")
            return

        self.running = True
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))

        # Schedule analysis for each timeframe
        interval_map = {
            "M5": 300,    # 5 minutes
            "M30": 1800,  # 30 minutes
            "H1": 3600,   # 1 hour
            "H4": 14400   # 4 hours
        }

        for timeframe in self.cfg['timeframes']:
            context.job_queue.run_repeating(
                self._market_analysis,
                interval=interval_map.get(timeframe, 3600),
                first=10,
                chat_id=update.effective_chat.id,
                name=f"analysis_{timeframe}"
            )

        await update.message.reply_text(
            f"🚀 Bot started analyzing:\n"
            f"Symbols: {', '.join(self.cfg['symbols'])}\n"
            f"Timeframes: {', '.join(self.cfg['timeframes'])}"
        )

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        if not self.running:
            await update.message.reply_text("⚠️ Bot isn't running!")
            return

        # Remove all jobs
        for timeframe in self.cfg['timeframes']:
            jobs = context.job_queue.get_jobs_by_name(f"analysis_{timeframe}")
            for job in jobs:
                job.schedule_removal()

        mt5.shutdown()
        self.running = False
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        
        await update.message.reply_text("🛑 Bot stopped successfully")

    def run(self):
        """Start the bot"""
        self.logger.info("Bot is now running...")
        self.application.run_polling()

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle all telegram.ext errors"""
        self.logger.error(f"Telegram error: {context.error}", exc_info=True)

if __name__ == "__main__":
    bot = VWAPMasterBot()
    bot.run()