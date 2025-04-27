import MetaTrader5 as mt5

def get_tradable_symbols():
    if not mt5.initialize():
        print("MT5 initialize() failed")
        return []
    
    try:
        symbols = mt5.symbols_get()
        print(f"Total symbols available: {len(symbols)}")
        
        tradable = []
        for s in symbols:
            # Check if trading is allowed (not disabled, not close-only, etc.)
            if s.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:  # Full trading allowed
                tradable.append(s.name)
        
        return sorted(tradable)
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    symbols = get_tradable_symbols()
    print("\nTradable Symbols:")
    print(symbols)