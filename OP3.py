#!/usr/bin/env python3
"""
Volatility Arbitrage Strategy - Main Execution Script

This script implements the volatility arbitrage strategy as a procedural flow.
"""

import numpy as np
from scipy.stats import norm
import time
import math
import logging

# Initialize services
s = requests.Session()
s.headers.update({'X-API-key': 'YQPBX6CUa'}) # Make sure you use YOUR API Key

# Configuration
ETF_TICKER = 'RTM'
OPTION_TICKERS = ['RTM48C', 'RTM49C', 'RTM50C', 'RTM48P', 'RTM49P', 'RTM50P']
RISK_FREE_RATE = 0.00
MAX_DELTA = 7000
MAX_OPTION_NET = 1000
MAX_OPTION_GROSS = 2500
CONTRACT_SIZE = 100
HEDGE_FREQUENCY = 30  # seconds
MISPRICING_THRESHOLD = 0.02  # 2%

# Global state
current_vol = None
next_vol_range = None
next_vol_mid = None
portfolio_delta = 0
option_positions = {ticker: 0 for ticker in OPTION_TICKERS}

# def configure_logging():
#     """Set up logging configuration"""
#     logging.basicConfig(
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         level=logging.INFO
#     )

def black_scholes_price(S, K, T, sigma, option_type):
    """Calculate option price using Black-Scholes"""
    d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'C':
        return S * norm.cdf(d1) - K * math.exp(-RISK_FREE_RATE * T) * norm.cdf(d2)
    else:  # put
        return K * math.exp(-RISK_FREE_RATE * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_delta(S, K, T, sigma, option_type):
    """Calculate option delta"""
    d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)

def black_scholes_vega(S, K, T, sigma, option_type):
    """Calculate option vega"""
    d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * norm.pdf(d1)

def black_scholes_implied_vol(S, K, T, price, option_type):
    """Calculate implied volatility using Black-Scholes model"""
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5  # Initial guess
    
    for _ in range(MAX_ITERATIONS):
        price_est = black_scholes_price(S, K, T, sigma, option_type)
        vega = black_scholes_vega(S, K, T, sigma, option_type)
        
        diff = price_est - price
        if abs(diff) < PRECISION:
            return sigma
            
        sigma = sigma - diff/vega
    
    return sigma

def get_latest_news(news_id):
    resp = s.get('http://localhost:9999/v1/news', params = {'limit': 50}) # default limit is 20
    if resp.ok:
        news_query = resp.json()

    return news_query[::-1][news_id]

def get_volatility_forecast():
    """Check for new volatility forecasts from analysts"""
    global current_vol, next_vol_range, next_vol_mid
    news = get_latest_news(-1)
    
    for item in news:
      if "volatility forecast" in item['headline'].lower():
          if "current week" in item['body'].lower():
              current_vol = float(item['body'].split('%')[0]) / 100
              print(f"Updated current volatility: {current_vol:.2%}")
              # logging.info(f"Updated current volatility: {current_vol:.2%}")
          elif "next week" in item['body'].lower():
              vol_range = item['body'].split('%')[0].split('-')
              next_vol_range = (float(vol_range[0])/100, float(vol_range[1])/100)
              next_vol_mid = sum(next_vol_range) / 2
              print(f"Updated next week volatility range: {next_vol_range[0]:.2%}-{next_vol_range[1]:.2%}")
                # logging.info(f"Updated next week volatility range: {next_vol_range[0]:.2%}-{next_vol_range[1]:.2%}")

def get_etf_position():
    resp = s.get ('http://localhost:9999/v1/securities')
    if resp.ok:
        book = resp.json()
        etf_position = book[0]['position']
        return etf_position

def get_tick():   
    resp = s.get('http://localhost:9999/v1/case')
    if resp.ok:
        case = resp.json()
        return case['tick'], case['status']

def place_order(ticker, volume, action, price=0):
    if price == 0:
        resp = s.post('http://localhost:9999/v1/orders', params = {'ticker': ticker, 'type': 'MARKET', 'quantity': volume, 'action': action})
    else:
        resp = s.post('http://localhost:9999/v1/orders', params = {'ticker': ticker, 'type': 'LIMIT', 'quantity': volume, 'price': price, 'action': action})
    return resp.json()

def calculate_portfolio_greeks(etf_price):
    """Calculate current portfolio Greeks"""
    global portfolio_delta
    portfolio_delta = 0
    portfolio_gamma = 0
    portfolio_vega = 0
    
    # ETF delta
    etf_position = get_etf_positionn(ETF_TICKER)
    portfolio_delta += etf_position  # ETF has delta of 1
    
    # Option Greeks
    tick, status = get_tick()
    days_remaining = tick / 15
    T = days_remaining / 240  # Fraction of month remaining
    
    for ticker, position in option_positions.items():
        if position == 0:
            continue
            
        strike = float(ticker[3:-1])
        option_type = ticker[-1]
        sigma = current_vol if current_vol else next_vol_mid
        
        position_size = position * CONTRACT_SIZE
        delta = black_scholes_delta(etf_price, strike, T, sigma, option_type)
        vega = black_scholes_vega(etf_price, strike, T, sigma, option_type)
        
        portfolio_delta += position_size * delta
        portfolio_vega += position_size * vega
    
    # logging.debug(f"Portfolio Greeks - Delta: {portfolio_delta:.0f}, Vega: {portfolio_vega:.2f}")
    return portfolio_gamma, portfolio_vega

def hedge_delta(etf_price):
    """Adjust ETF position to maintain delta neutrality"""
    current_etf_position = get_etf_position(ETF_TICKER)
    target_etf_position = -round(portfolio_delta)  # Aim for zero total delta
    target_etf_position = max(-50000, min(50000, target_etf_position))
    
    trade_size = target_etf_position - current_etf_position
    if abs(trade_size) > 0:
        action = 'BUY' if trade_size > 0 else 'SELL'
        place_order(ETF_TICKER, trade_size, action, etf_price)
        print(f"Delta hedge: {action} {abs(trade_size)} shares at {etf_price:.2f}")
        # logging.info(f"Delta hedge: {action} {abs(trade_size)} shares at {etf_price:.2f}")

def get_bid_ask(ticker):
    payload = {'ticker': ticker}
    resp = s.get ('http://localhost:9999/v1/securities/book', params = payload)
    if resp.ok:
        book = resp.json()
        bid_side_book = book['bids']
        ask_side_book = book['asks']
        
        bid_prices_book = [item["price"] for item in bid_side_book]
        ask_prices_book = [item['price'] for item in ask_side_book]
        
        best_bid_price = bid_prices_book[0]
        best_ask_price = ask_prices_book[0]
  
        return best_bid_price, best_ask_price

def find_mispriced_options(etf_price):
    """Identify options where implied vol differs significantly from forecast"""
    mispriced = []
    # Option Greeks
    tick, status = get_tick()
    days_remaining = tick / 15
    T = days_remaining / 240  # Fraction of month remaining
    forecast_vol = current_vol if current_vol else next_vol_mid
    
    if forecast_vol is None:
        return mispriced
    
    for ticker in OPTION_TICKERS:
        bid, ask = get_bid_ask(ticker)
        if not bid or not ask:
            continue
            
        # bid = book['bids'][0]['price']
        # ask = book['asks'][0]['price']
      
        mid_price = (bid + ask) / 2
        strike = float(ticker[3:-1])
        option_type = ticker[-1]
        
        iv = black_scholes_implied_vol(etf_price, strike, T, mid_price, option_type)
        mispricing = iv - forecast_vol
        
        if abs(mispricing) > MISPRICING_THRESHOLD:
            mispriced.append({
                'ticker': ticker,
                'iv': iv,
                'forecast_vol': forecast_vol,
                'mispricing': mispricing,
                'price': mid_price,
                'bid': bid,
                'ask': ask,
                'type': option_type
            })
    
    return sorted(mispriced, key=lambda x: abs(x['mispricing']), reverse=True)

def execute_option_trades(mispriced_options):
    """Execute trades based on mispricing opportunities"""
    for option in mispriced_options:
        ticker = option['ticker']
        current_position = option_positions[ticker]
        max_trade = min(100, MAX_OPTION_NET - abs(current_position))
        
        if max_trade <= 0:
            continue
            
        if option['mispricing'] > 0:  # Overpriced - sell
            action = 'SELL'
            price = option['bid']
        else:  # Underpriced - buy
            action = 'BUY'
            price = option['ask']
          
        place_order(ticker, max_trade, action, price)
        # api.submit_order({
        #     'ticker': ticker,
        #     'type': 'LIMIT',
        #     'price': price,
        #     'quantity': max_trade,
        #     'action': action
        # })
        
        option_positions[ticker] += max_trade if action == 'BUY' else -max_trade
        print(f"Option trade: {action} {max_trade} {ticker} at {price:.2f}")
        # logging.info(f"Option trade: {action} {max_trade} {ticker} at {price:.2f}")

def close_all_positions():
    """Close all positions at end of trading period"""
    # Close options
    for ticker, position in option_positions.items():
        if position != 0:
            action = 'SELL' if position > 0 else 'BUY'
            place_order(ticker, abs(position), action)
            # api.submit_order({
            #     'ticker': ticker,
            #     'type': 'MARKET',
            #     'quantity': abs(position),
            #     'action': action
            # })
            # logging.info(f"Closing position: {action} {abs(position)} {ticker}")
    
    # Close ETF
    etf_position = get_etf_position(ETF_TICKER)
    if etf_position != 0:
        action = 'SELL' if etf_position > 0 else 'BUY'
        place_order(ticker, abs(position), action)
        # api.submit_order({
        #     'ticker': ETF_TICKER,
        #     'type': 'MARKET',
        #     'quantity': abs(etf_position),
        #     'action': action
        # })
        print(f"Closing ETF: {action} {abs(etf_position)} shares")
        # logging.info(f"Closing ETF: {action} {abs(etf_position)} shares")

def run_strategy():
    """Main strategy execution loop"""
    last_hedge_time, status = get_tick()
  
    # # Check case status
    # tick, status = get_tick()
    
    while  status == 'ACTIVE' or tick <= 295:
        try:
            # # Check case status
            # tick, status = get_tick()
            # if status != 'ACTIVE' or tick >= 595:
            #     logging.info("Case ending - closing all positions")
            #     close_all_positions()
            #     break
            
            # Get market data
            etf_bid, etf_ask = get_bid_ask(ETF_TICKER)
            etf_price = (etf_bid + etf_ask) / 2
            
            # Update volatility forecasts
            get_volatility_forecast()
            
            # Only trade if we have volatility forecasts
            if current_vol or next_vol_mid:
                # Find and trade mispriced options
                mispriced_options = find_mispriced_options(etf_price)
                if mispriced_options:
                    execute_option_trades(mispriced_options)
                
                # Recalculate Greeks
                calculate_portfolio_greeks(etf_price)
                
                # Periodic delta hedging
                current_time, status = get_tick()
                if current_time - last_hedge_time > HEDGE_FREQUENCY or abs(portfolio_delta) > MAX_DELTA * 0.8:
                    hedge_delta(etf_price)
                    last_hedge_time = current_time
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Strategy error: {str(e)}")
            # logging.error(f"Strategy error: {str(e)}")
            time.sleep(1)

if __name__ == "__main__":
    # configure_logging()
    # logging.info("Starting Volatility Arbitrage Strategy")
    print("Starting Volatility Arbitrage Strategy")
    try:
        run_strategy()
    except KeyboardInterrupt:
        # logging.info("Received interrupt - shutting down")
        print("Received interrupt - shutting down")
        close_all_positions()
    except Exception as e:
        # logging.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
    finally:
        # logging.info("Strategy terminated")
        print("Strategy terminated")
