import os

print("="*60)
print("🔍 ENVIRONMENT VARIABLES DEBUG")
print("="*60)

# Check करतो की variables set आहेत का
vars_to_check = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID", 
    "DHAN_CLIENT_ID",
    "DHAN_ACCESS_TOKEN",
    "CEREBRAS_API_KEY",
    "HYPERBOLIC_API_KEY",
    "DEEPSEEK_API_KEY"
]

all_good = True

for var_name in vars_to_check:
    value = os.getenv(var_name)
    
    if value:
        # First 10 chars show करतो (security साठी)
        masked_value = value[:10] + "..." if len(value) > 10 else value
        print(f"✅ {var_name}: {masked_value}")
    else:
        print(f"❌ {var_name}: NOT SET")
        all_good = False

print("="*60)

if all_good:
    print("🎉 All environment variables are SET!")
    print("✅ You can run the bot now!")
else:
    print("⚠️  Some variables are MISSING!")
    print("\n🔧 Railway/Render मध्ये add करा:")
    print("   Settings → Environment Variables → Add Variable")
    
print("="*60)
