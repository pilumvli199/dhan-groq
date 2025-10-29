"""
🔍 NIFTY 50 CSV PARSER - FINAL SOLUTION
Downloads CSV and finds EXACT Security IDs for NIFTY 50 options
"""

import requests
import csv
import io
from collections import Counter
from datetime import datetime

print("="*80)
print("🔍 NIFTY 50 SECURITY ID FINDER - CSV METHOD")
print("="*80)

# Download CSV
csv_url = "https://images.dhan.co/api-data/api-scrip-master.csv"

print(f"\n📥 Downloading CSV...")

try:
    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()
    
    print(f"✅ Downloaded ({len(response.content)} bytes)")
    
    # Parse CSV
    csv_content = response.content.decode('utf-8')
    csv_reader = csv.DictReader(io.StringIO(csv_content))
    
    print("\n🔍 Searching for NIFTY 50 derivatives...")
    print("-"*80)
    
    # Collect all NIFTY derivatives
    nifty_options = []
    nifty_futures = []
    nifty_index = []
    
    for row in csv_reader:
        try:
            trading_symbol = row.get('SEM_TRADING_SYMBOL', '').strip()
            segment = row.get('SEM_SEGMENT', '')
            exchange = row.get('SEM_EXM_EXCH_ID', '')
            instrument = row.get('SEM_INSTRUMENT_NAME', '')
            security_id = row.get('SEM_SMST_SECURITY_ID', '')
            expiry = row.get('SEM_EXPIRY_DATE', '')
            strike = row.get('SEM_STRIKE_PRICE', '')
            option_type = row.get('SEM_OPTION_TYPE', '')
            
            # Skip if no security ID
            if not security_id:
                continue
            
            # INDEX (Spot)
            if segment == 'I' and exchange == 'NSE':
                if trading_symbol == 'NIFTY' or 'NIFTY 50' in row.get('SEM_CUSTOM_SYMBOL', ''):
                    nifty_index.append({
                        'security_id': security_id,
                        'symbol': trading_symbol,
                        'instrument': instrument
                    })
            
            # DERIVATIVES (F&O)
            elif segment == 'D' and exchange == 'NSE':
                # Must start with "NIFTY " or be exactly "NIFTY"
                # But NOT "NIFTYBANK" or "NIFTY BANK"
                if trading_symbol.startswith('NIFTY ') or trading_symbol == 'NIFTY':
                    if 'BANK' not in trading_symbol:
                        entry = {
                            'security_id': security_id,
                            'symbol': trading_symbol,
                            'instrument': instrument,
                            'expiry': expiry,
                            'strike': strike,
                            'option_type': option_type
                        }
                        
                        if instrument == 'OPTIDX':  # Options
                            nifty_options.append(entry)
                        elif instrument == 'FUTIDX':  # Futures
                            nifty_futures.append(entry)
        
        except Exception as e:
            continue
    
    print(f"✅ Found:")
    print(f"   Index entries: {len(nifty_index)}")
    print(f"   Futures: {len(nifty_futures)}")
    print(f"   Options: {len(nifty_options)}")
    
    # Display INDEX
    print("\n" + "="*80)
    print("📊 NIFTY 50 INDEX (SPOT PRICE)")
    print("="*80)
    
    if nifty_index:
        for idx in nifty_index:
            print(f"Security ID: {idx['security_id']}")
            print(f"Symbol: {idx['symbol']}")
            print(f"Instrument: {idx['instrument']}")
            print("-"*80)
    else:
        print("❌ No index entries found!")
    
    # Display FUTURES
    print("\n" + "="*80)
    print("🔮 NIFTY 50 FUTURES")
    print("="*80)
    
    if nifty_futures:
        # Show unique security IDs
        future_ids = Counter([f['security_id'] for f in nifty_futures])
        print(f"\nUnique Security IDs: {list(future_ids.keys())}")
        
        # Show sample
        for fut in nifty_futures[:3]:
            print(f"\nSecurity ID: {fut['security_id']}")
            print(f"Symbol: {fut['symbol']}")
            print(f"Expiry: {fut['expiry']}")
    else:
        print("❌ No futures found!")
    
    # Display OPTIONS
    print("\n" + "="*80)
    print("🎯 NIFTY 50 OPTIONS (MOST IMPORTANT!)")
    print("="*80)
    
    if nifty_options:
        # Count by security ID
        option_ids = Counter([o['security_id'] for o in nifty_options])
        
        print(f"\n📊 Security ID Distribution:")
        for sec_id, count in option_ids.most_common():
            print(f"   Security ID {sec_id}: {count} contracts")
        
        # Get most common (underlying)
        underlying_id = option_ids.most_common(1)[0][0]
        
        print(f"\n✅ NIFTY 50 Options Underlying Security ID: {underlying_id}")
        
        # Show sample contracts with this ID
        sample_options = [o for o in nifty_options if o['security_id'] == underlying_id][:5]
        
        print(f"\n📋 Sample Option Contracts (Security ID: {underlying_id}):")
        print("-"*80)
        
        for opt in sample_options:
            print(f"Symbol: {opt['symbol']}")
            print(f"Expiry: {opt['expiry']}")
            print(f"Strike: {opt['strike']}")
            print(f"Type: {opt['option_type']}")
            print("-"*40)
        
        # Check expiry types
        expiries = sorted(set([o['expiry'] for o in sample_options if o['expiry']]))
        if expiries:
            print(f"\n📅 Sample Expiries: {expiries[:5]}")
            
            # Check if weekly
            today = datetime.now().date()
            weekly_count = 0
            monthly_count = 0
            
            for exp_str in expiries[:10]:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    # Tuesday = 1
                    if exp_date.weekday() == 1:
                        weekly_count += 1
                    # Last Thursday/Tuesday of month
                    if exp_date.day > 20:
                        monthly_count += 1
                except:
                    pass
            
            print(f"\n📊 Expiry Analysis (first 10):")
            print(f"   Weekly-like (Tuesdays): {weekly_count}")
            print(f"   Monthly-like (end of month): {monthly_count}")
    
    else:
        print("❌ No options found!")
        print("\n⚠️ This means NIFTY 50 options might not be in CSV")
        print("   OR they use a different naming convention")
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("🎯 FINAL ANSWER - USE THESE IDs:")
    print("="*80)
    
    if nifty_index:
        print(f"✅ INDEX_SECURITY_ID = {nifty_index[0]['security_id']}  # For spot price")
        print(f"   Segment: IDX_I")
    
    if nifty_options:
        underlying_id = Counter([o['security_id'] for o in nifty_options]).most_common(1)[0][0]
        print(f"✅ FNO_SECURITY_ID = {underlying_id}  # For options")
        print(f"   Segment: NSE_FNO")
    elif nifty_futures:
        future_id = Counter([f['security_id'] for f in nifty_futures]).most_common(1)[0][0]
        print(f"⚠️ FNO_SECURITY_ID = {future_id}  # From futures (fallback)")
        print(f"   Segment: NSE_FNO")
        print(f"   WARNING: This is from futures, options might have different ID")
    
    print("="*80)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 SCAN COMPLETE")
