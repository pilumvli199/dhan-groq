"""
🧪 NIFTY 50 API TESTER
Tests different Security IDs to find correct one for NIFTY 50 options
"""

import requests
import json
import os
from datetime import datetime

# Your credentials
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "1108547103")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "your_token_here")

DHAN_API_BASE = "https://api.dhan.co"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"

headers = {
    'access-token': DHAN_ACCESS_TOKEN,
    'client-id': DHAN_CLIENT_ID,
    'Content-Type': 'application/json'
}

print("="*80)
print("🧪 NIFTY 50 API TESTER - FINDING CORRECT SECURITY ID")
print("="*80)

# Test different possible Security IDs for NIFTY 50
# Based on common patterns in Dhan API
test_ids = [
    13,    # Index spot
    25,    # Common F&O ID
    26,    # Another common ID
    1333,  # Pattern based
    2513,  # Pattern based
]

print(f"\n📋 Testing {len(test_ids)} possible Security IDs for NIFTY 50 F&O...")
print("-"*80)

successful_ids = []

for test_id in test_ids:
    print(f"\n🔍 Testing Security ID: {test_id}")
    print("   Segment: NSE_FNO")
    
    payload = {
        "UnderlyingScrip": test_id,
        "UnderlyingSeg": "NSE_FNO"
    }
    
    try:
        # Test 1: Get expiry list
        response = requests.post(
            DHAN_EXPIRY_LIST_URL,
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 'success' and data.get('data'):
                expiries = data['data']
                print(f"   ✅ Expiry API Success!")
                print(f"   📅 Found {len(expiries)} expiries: {expiries[:3]}")
                
                # Test 2: Get option chain for first expiry
                if expiries:
                    test_expiry = expiries[0]
                    print(f"\n   🔍 Testing option chain for expiry: {test_expiry}")
                    
                    oc_payload = {
                        "UnderlyingScrip": test_id,
                        "UnderlyingSeg": "NSE_FNO",
                        "Expiry": test_expiry
                    }
                    
                    oc_response = requests.post(
                        DHAN_OPTION_CHAIN_URL,
                        json=oc_payload,
                        headers=headers,
                        timeout=10
                    )
                    
                    if oc_response.status_code == 200:
                        oc_data = oc_response.json()
                        
                        if oc_data.get('data') and oc_data['data'].get('oc'):
                            strikes = list(oc_data['data']['oc'].keys())
                            
                            if strikes:
                                # Convert to float and find range
                                strike_floats = []
                                for s in strikes[:10]:  # Sample first 10
                                    try:
                                        strike_floats.append(float(s))
                                    except:
                                        pass
                                
                                if strike_floats:
                                    min_strike = min(strike_floats)
                                    max_strike = max(strike_floats)
                                    
                                    print(f"   ✅ Option Chain Success!")
                                    print(f"   📊 Total strikes: {len(strikes)}")
                                    print(f"   📊 Strike range (sample): {min_strike:.0f} to {max_strike:.0f}")
                                    
                                    # Check if strikes are in NIFTY 50 range (20000-30000)
                                    if 20000 <= min_strike <= 30000:
                                        print(f"   🎯 STRIKE RANGE MATCHES NIFTY 50!")
                                        
                                        # Get more details
                                        sample_strike = strikes[0]
                                        strike_data = oc_data['data']['oc'][sample_strike]
                                        
                                        ce_data = strike_data.get('ce', {})
                                        pe_data = strike_data.get('pe', {})
                                        
                                        print(f"\n   📈 Sample Strike: {float(sample_strike):.0f}")
                                        print(f"      CE OI: {ce_data.get('oi', 0):,}")
                                        print(f"      PE OI: {pe_data.get('oi', 0):,}")
                                        print(f"      CE LTP: ₹{ce_data.get('ltp', 0):.2f}")
                                        print(f"      PE LTP: ₹{pe_data.get('ltp', 0):.2f}")
                                        
                                        successful_ids.append({
                                            'security_id': test_id,
                                            'expiries': expiries,
                                            'total_strikes': len(strikes),
                                            'strike_range': f"{min_strike:.0f}-{max_strike:.0f}"
                                        })
                                        
                                        print(f"\n   ✅✅✅ THIS IS THE CORRECT SECURITY ID! ✅✅✅")
                                    
                                    elif 40000 <= min_strike <= 60000:
                                        print(f"   ⚠️ Strike range matches BANK NIFTY (not NIFTY 50)")
                                    elif min_strike < 10000:
                                        print(f"   ⚠️ Strike range too low - might be different index")
                                    else:
                                        print(f"   ⚠️ Strike range doesn't match NIFTY 50")
                        else:
                            print(f"   ❌ Option chain has no strikes")
                    else:
                        print(f"   ❌ Option chain API failed: {oc_response.status_code}")
            else:
                print(f"   ❌ No expiries found")
        else:
            print(f"   ❌ API failed: {response.status_code}")
            if response.status_code == 401:
                print(f"   ⚠️ Authentication error - check your access token")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("-"*80)

# Summary
print("\n" + "="*80)
print("📋 TEST SUMMARY")
print("="*80)

if successful_ids:
    print(f"✅ Found {len(successful_ids)} working Security ID(s):\n")
    
    for result in successful_ids:
        print(f"🎯 Security ID: {result['security_id']}")
        print(f"   Expiries: {len(result['expiries'])} available")
        print(f"   First expiry: {result['expiries'][0]}")
        print(f"   Strikes: {result['total_strikes']} total")
        print(f"   Strike range: {result['strike_range']}")
        print()
    
    print("="*80)
    print("💡 USE THIS IN YOUR BOT:")
    print("="*80)
    print(f"INDEX_SECURITY_ID = 13  # For spot price")
    print(f"FNO_SECURITY_ID = {successful_ids[0]['security_id']}  # For options")
    print(f"FNO_SEGMENT = 'NSE_FNO'")
    print("="*80)
else:
    print("❌ No working Security ID found!")
    print("\n💡 Possible solutions:")
    print("   1. Download instrument CSV and search manually")
    print("   2. Check Dhan API documentation")
    print("   3. Contact Dhan support")
    print("\n📥 CSV URL: https://images.dhan.co/api-data/api-scrip-master.csv")

print("\n" + "="*80)
print("🏁 TEST COMPLETE")
print("="*80)
