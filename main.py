# test_api.py - API तपासण्यासाठी
import requests
import json
from datetime import datetime, timedelta
import pytz
import os

headers = {
    'access-token': os.getenv("DHAN_ACCESS_TOKEN"),
    'client-id': os.getenv("DHAN_CLIENT_ID"),
    'Content-Type': 'application/json'
}

ist = pytz.timezone('Asia/Kolkata')
to_date = datetime.now(ist)
from_date = to_date - timedelta(days=7)

# Test with RELIANCE
payload = {
    "securityId": "2885",
    "exchangeSegment": "NSE_EQ",
    "instrument": "EQUITY",
    "interval": "5",
    "fromDate": from_date.strftime("%Y-%m-%d"),
    "toDate": to_date.strftime("%Y-%m-%d")
}

print(f"⏰ IST Time: {datetime.now(ist).strftime('%H:%M:%S')}")
print(f"📅 Date Range: {from_date.date()} to {to_date.date()}")
print(f"📤 Request: {json.dumps(payload, indent=2)}")

response = requests.post(
    "https://api.dhan.co/v2/charts/intraday",
    json=payload,
    headers=headers,
    timeout=15
)

print(f"\n📥 Status: {response.status_code}")
data = response.json()
print(f"📊 Response Keys: {list(data.keys())}")

if 'open' in data:
    print(f"📈 Data points: {len(data['open'])}")
    if len(data['open']) > 0:
        print(f"✅ First candle: Open={data['open'][0]}, Close={data['close'][0]}")
    else:
        print(f"❌ Empty arrays!")
else:
    print(f"❌ No 'open' key in response")
    print(f"📋 Full response: {json.dumps(data, indent=2)}")
