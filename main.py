"""
Dhan API Debug & Test Script
Test करण्यासाठी हे script चालवा
"""

import requests
import os
import csv
import io
import json
from datetime import datetime, timedelta
import pytz

# Configuration
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"

headers = {
    'access-token': DHAN_ACCESS_TOKEN,
    'client-id': DHAN_CLIENT_ID,
    'Content-Type': 'application/json'
}

print("="*70)
print("🧪 DHAN API DEBUG TESTER")
print("="*70)

# Test 1: Download instruments CSV
print("\n📥 TEST 1: Downloading instruments CSV...")
try:
    response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
    if response.status_code == 200:
        print(f"✅ CSV downloaded successfully ({len(response.text)} bytes)")
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(response.text))
        all_rows = list(csv_reader)
        print(f"📊 Total instruments: {len(all_rows)}")
        
        if all_rows:
            print(f"📋 CSV Headers: {list(all_rows[0].keys())}")
            print(f"\n🔍 Sample row (first NSE equity):")
            for row in all_rows[:100]:
                if row.get('SEM_SEGMENT') == 'E' and row.get('SEM_EXM_EXCH_ID') == 'NSE':
                    print(json.dumps(row, indent=2))
                    break
        
        # Find RELIANCE
        print("\n🔍 TEST 2: Finding RELIANCE...")
        reliance_found = False
        for row in all_rows:
            if (row.get('SEM_SEGMENT') == 'E' and 
                row.get('SEM_EXM_EXCH_ID') == 'NSE' and
                row.get('SEM_TRADING_SYMBOL') == 'RELIANCE'):
                print(f"✅ RELIANCE found!")
                print(f"Security ID: {row.get('SEM_SMST_SECURITY_ID')}")
                print(f"Segment: {row.get('SEM_SEGMENT')}")
                print(f"Exchange: {row.get('SEM_EXM_EXCH_ID')}")
                print(f"Trading Symbol: {row.get('SEM_TRADING_SYMBOL')}")
                reliance_id = row.get('SEM_SMST_SECURITY_ID')
                reliance_found = True
                break
        
        if not reliance_found:
            print("❌ RELIANCE not found in CSV")
            exit(1)
        
        # Test 3: Fetch candles for RELIANCE
        print("\n📊 TEST 3: Fetching candles for RELIANCE...")
        
        ist = pytz.timezone('Asia/Kolkata')
        to_date = datetime.now(ist)
        from_date = to_date - timedelta(days=7)
        
        # Try different payload formats
        payloads_to_test = [
            {
                "securityId": reliance_id,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "expiryCode": 0,
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            },
            {
                "securityId": reliance_id,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            },
            {
                "securityId": reliance_id,
                "exch_seg": "NSE_EQ",
                "instrument": "EQUITY",
                "interval": "5",
                "from_date": from_date.strftime("%Y-%m-%d"),
                "to_date": to_date.strftime("%Y-%m-%d")
            }
        ]
        
        for i, payload in enumerate(payloads_to_test, 1):
            print(f"\n🔄 Attempt {i}: Testing payload...")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                DHAN_INTRADAY_URL,
                json=payload,
                headers=headers,
                timeout=15
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS!")
                print(f"Response keys: {list(data.keys())}")
                
                # Check for different timestamp field names
                timestamp_fields = ['timestamp', 'start_time', 'time', 'date']
                timestamp_found = None
                for field in timestamp_fields:
                    if field in data:
                        timestamp_found = field
                        break
                
                if timestamp_found:
                    print(f"📊 Candles found!")
                    print(f"Timestamp field: {timestamp_found}")
                    print(f"Number of candles: {len(data.get(timestamp_found, []))}")
                    if data.get('open'):
                        print(f"Sample data:")
                        print(f"  Open[0]: {data['open'][0] if data['open'] else 'N/A'}")
                        print(f"  Close[0]: {data['close'][0] if data['close'] else 'N/A'}")
                        print(f"  High[0]: {data['high'][0] if data['high'] else 'N/A'}")
                        print(f"  Low[0]: {data['low'][0] if data['low'] else 'N/A'}")
                    break
                else:
                    print(f"⚠️ No timestamp field found")
                    print(f"Full response: {json.dumps(data, indent=2)[:500]}...")
            else:
                print(f"❌ Error: {response.text[:200]}")
        
        # Test 4: Check expiry
        print("\n📅 TEST 4: Checking expiry for RELIANCE...")
        expiry_payload = {
            "UnderlyingScrip": int(reliance_id),
            "UnderlyingSeg": "NSE_EQ"
        }
        print(f"Payload: {json.dumps(expiry_payload, indent=2)}")
        
        response = requests.post(
            DHAN_EXPIRY_LIST_URL,
            json=expiry_payload,
            headers=headers,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            if data.get('data'):
                print(f"✅ Expiry found: {data['data'][0]}")
            else:
                print(f"ℹ️ No expiry (RELIANCE may not have stock options)")
        else:
            print(f"❌ Error: {response.text}")
        
    else:
        print(f"❌ Failed to download CSV: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ TEST COMPLETE!")
print("="*70)
