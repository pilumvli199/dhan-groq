"""
🔍 NIFTY 50 Security ID Finder
Downloads Dhan's instrument CSV and finds correct Security IDs
"""

import requests
import csv
import io

def find_nifty_security_ids():
    """
    Download Dhan instrument CSV and find NIFTY 50 Security IDs
    """
    
    print("="*70)
    print("🔍 FINDING NIFTY 50 SECURITY IDs FROM DHAN CSV")
    print("="*70)
    
    # Download CSV
    csv_url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    
    print(f"\n📥 Downloading: {csv_url}")
    
    try:
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()
        
        print(f"✅ Downloaded successfully ({len(response.content)} bytes)")
        
        # Parse CSV
        csv_content = response.content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        print("\n🔍 Searching for NIFTY 50 entries...")
        print("-"*70)
        
        nifty_entries = []
        
        for row in csv_reader:
            # Check if this is NIFTY 50 or NIFTY related
            symbol = row.get('SEM_CUSTOM_SYMBOL', '').upper()
            trading_symbol = row.get('SEM_TRADING_SYMBOL', '').upper()
            segment = row.get('SEM_SEGMENT', '')
            exch = row.get('SEM_EXM_EXCH_ID', '')
            
            # Look for NIFTY (not BANK NIFTY)
            if 'NIFTY' in symbol or 'NIFTY' in trading_symbol:
                if 'BANK' not in symbol and 'BANK' not in trading_symbol:
                    if 'NIFTY 50' in symbol or trading_symbol == 'NIFTY':
                        nifty_entries.append({
                            'security_id': row.get('SEM_SMST_SECURITY_ID', ''),
                            'symbol': symbol,
                            'trading_symbol': trading_symbol,
                            'segment': segment,
                            'exchange': exch,
                            'instrument': row.get('SEM_INSTRUMENT_NAME', ''),
                            'expiry_date': row.get('SEM_EXPIRY_DATE', ''),
                            'lot_size': row.get('SEM_LOT_UNITS', '')
                        })
        
        print(f"✅ Found {len(nifty_entries)} NIFTY 50 entries\n")
        
        # Group by segment
        spot_entries = []
        fno_entries = []
        
        for entry in nifty_entries:
            if entry['segment'] == 'I':  # Index segment (Spot)
                spot_entries.append(entry)
            elif entry['segment'] == 'D':  # Derivatives (F&O)
                fno_entries.append(entry)
        
        # Display Spot entries
        print("📊 NIFTY 50 SPOT (Index) - Segment: IDX_I")
        print("="*70)
        
        if spot_entries:
            for entry in spot_entries[:5]:  # Show first 5
                print(f"Security ID: {entry['security_id']}")
                print(f"Symbol: {entry['symbol']}")
                print(f"Trading Symbol: {entry['trading_symbol']}")
                print(f"Exchange: {entry['exchange']}")
                print(f"Instrument: {entry['instrument']}")
                print("-"*70)
        else:
            print("❌ No SPOT entries found!")
        
        # Display F&O entries
        print("\n🎯 NIFTY 50 F&O (Options) - Segment: NSE_FNO")
        print("="*70)
        
        if fno_entries:
            # Group by expiry
            expiry_groups = {}
            for entry in fno_entries:
                expiry = entry['expiry_date']
                if expiry not in expiry_groups:
                    expiry_groups[expiry] = []
                expiry_groups[expiry].append(entry)
            
            # Show unique security IDs
            unique_ids = set()
            for entry in fno_entries[:10]:  # First 10
                unique_ids.add(entry['security_id'])
                print(f"Security ID: {entry['security_id']}")
                print(f"Symbol: {entry['symbol']}")
                print(f"Trading Symbol: {entry['trading_symbol']}")
                print(f"Expiry: {entry['expiry_date']}")
                print(f"Lot Size: {entry['lot_size']}")
                print("-"*70)
            
            print(f"\n📌 Unique Security IDs in F&O: {unique_ids}")
            
        else:
            print("❌ No F&O entries found!")
        
        # Summary
        print("\n" + "="*70)
        print("📋 SUMMARY")
        print("="*70)
        
        if spot_entries:
            spot_id = spot_entries[0]['security_id']
            print(f"✅ NIFTY 50 SPOT Security ID: {spot_id}")
            print(f"   Segment: IDX_I")
        
        if fno_entries:
            # Get most common security ID in F&O
            from collections import Counter
            fno_ids = [e['security_id'] for e in fno_entries]
            most_common = Counter(fno_ids).most_common(1)[0][0]
            print(f"✅ NIFTY 50 F&O Security ID: {most_common}")
            print(f"   Segment: NSE_FNO")
        
        print("\n" + "="*70)
        print("💡 USE THESE IDs IN YOUR BOT:")
        print("="*70)
        if spot_entries and fno_entries:
            print(f"INDEX_SECURITY_ID = {spot_entries[0]['security_id']}  # For spot price")
            print(f"FNO_SECURITY_ID = {Counter([e['security_id'] for e in fno_entries]).most_common(1)[0][0]}  # For options")
        print("="*70)
        
        return {
            'spot': spot_entries,
            'fno': fno_entries
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = find_nifty_security_ids()
