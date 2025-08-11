#!/usr/bin/env python3
"""
OpenAI API Accessibility Test
Tests if the provided API key can successfully connect to OpenAI's API
"""

import os
import sys
from openai import OpenAI

# Import configuration
try:
    from config.config import OPENAI_API_KEY
except ImportError:
    print("❌ Configuration file not found. Please copy config_template.py to config.py and add your API keys.")
    sys.exit(1)

def test_openai_access(api_key):
    """Test OpenAI API accessibility with the provided key"""
    
    # Initialize the client
    client = OpenAI(api_key=api_key)
    
    try:
        print("🔍 Testing OpenAI API accessibility...")
        print(f"🔑 API Key: {api_key[:10]}...{api_key[-10:]}")
        print()
        
        # Test 1: List models (lightweight check)
        print("📋 Test 1: Listing available models...")
        models = client.models.list()
        model_names = [model.id for model in models.data]
        print(f"✅ Successfully retrieved {len(model_names)} models")
        
        # Show GPT models specifically
        gpt_models = [m for m in model_names if 'gpt' in m.lower()]
        print(f"🤖 Available GPT models: {', '.join(sorted(gpt_models)[:10])}")
        if len(gpt_models) > 10:
            print(f"   ... and {len(gpt_models) - 10} more GPT models")
        print()
        
        # Test 2: Simple completion test
        print("🤖 Test 2: Testing completion API with a joke request...")
        model_used = "gpt-3.5-turbo"
        print(f"   Using model: {model_used}")
        response = client.chat.completions.create(
            model=model_used,
            messages=[
                {"role": "user", "content": "Tell me a joke"}
            ],
            max_tokens=100
        )
        
        completion_text = response.choices[0].message.content
        print(f"✅ Completion successful: {completion_text}")
        print()
        
        # Display usage info
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"📊 Token usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
        print()
        
        # Test 3: Check account balance/credits
        print("💳 Test 3: Checking account balance and usage...")
        try:
            # Alternative approach using HTTP requests to billing API
            import requests
            import json
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Try to get subscription info
            try:
                sub_response = requests.get(
                    'https://api.openai.com/v1/dashboard/billing/subscription',
                    headers=headers,
                    timeout=10
                )
                if sub_response.status_code == 200:
                    sub_data = sub_response.json()
                    print(f"✅ Account Type: {sub_data.get('plan', {}).get('title', 'Unknown')}")
                    print(f"   Status: {sub_data.get('status', 'Unknown')}")
                else:
                    print(f"ℹ️  Subscription info: HTTP {sub_response.status_code}")
            except Exception as e:
                print(f"ℹ️  Subscription check failed: {type(e).__name__}")
            
            # Try to get usage info
            try:
                import datetime
                current_date = datetime.datetime.now()
                start_date = current_date.replace(day=1).strftime('%Y-%m-%d')
                end_date = current_date.strftime('%Y-%m-%d')
                
                usage_response = requests.get(
                    f'https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date}&end_date={end_date}',
                    headers=headers,
                    timeout=10
                )
                
                if usage_response.status_code == 200:
                    usage_data = usage_response.json()
                    total_usage = usage_data.get('total_usage', 0) / 100
                    print(f"💰 Current month usage: ${total_usage:.4f}")
                else:
                    print(f"ℹ️  Usage info: HTTP {usage_response.status_code}")
            except Exception as e:
                print(f"ℹ️  Usage check failed: {type(e).__name__}")
            
            # Try to get credit grants
            try:
                credits_response = requests.get(
                    'https://api.openai.com/v1/dashboard/billing/credit_grants',
                    headers=headers,
                    timeout=10
                )
                
                if credits_response.status_code == 200:
                    credits_data = credits_response.json()
                    grants = credits_data.get('data', [])
                    
                    total_credits = 0
                    active_grants = 0
                    
                    for grant in grants:
                        grant_amount = grant.get('grant_amount', 0) / 100
                        used_amount = grant.get('used_amount', 0) / 100
                        remaining = grant_amount - used_amount
                        
                        if remaining > 0:
                            total_credits += remaining
                            active_grants += 1
                            expires_at = grant.get('expires_at', 'No expiration')
                            print(f"   Active Credit: ${remaining:.4f} (Expires: {expires_at})")
                    
                    if active_grants > 0:
                        print(f"💎 Total remaining credits: ${total_credits:.4f}")
                    else:
                        print("ℹ️  No active credit grants found")
                else:
                    print(f"ℹ️  Credit grants info: HTTP {credits_response.status_code}")
                    if credits_response.status_code == 401:
                        print("   This might be a pay-as-you-go account without credit grants")
            except Exception as e:
                print(f"ℹ️  Credits check failed: {type(e).__name__}")
                
        except Exception as billing_error:
            print(f"ℹ️  Billing API access not available: {type(billing_error).__name__}")
            print("   Note: Billing information might require special API permissions")
        
        print()
        print("🎉 OpenAI API is fully accessible!")
        return True
        
    except Exception as e:
        print(f"❌ Error accessing OpenAI API:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Provide specific guidance based on error type
        if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
            print("💡 This appears to be an authentication issue. Please check:")
            print("   - API key is correct and not expired")
            print("   - Account has sufficient credits")
            print("   - API key has necessary permissions")
        elif "rate" in str(e).lower() or "quota" in str(e).lower():
            print("💡 This appears to be a rate limit or quota issue.")
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            print("💡 This appears to be a network connectivity issue.")
        
        return False

if __name__ == "__main__":
    # API key from configuration
    api_key = OPENAI_API_KEY
    
    success = test_openai_access(api_key)
    sys.exit(0 if success else 1)
