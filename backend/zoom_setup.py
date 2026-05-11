"""
zoom_setup.py
─────────────
Complete setup guide + test tool for Zoom Webhook integration.

Run this AFTER starting the EVA server:
    python zoom_setup.py          # show setup guide
    python zoom_setup.py --test   # send a test webhook
    python zoom_setup.py --ngrok  # check ngrok URL
"""

import sys
import json
import hmac
import hashlib
import requests
import argparse
from datetime import datetime

EVA_SERVER = "http://localhost:8000"


def get_ngrok_url():
    """Get current ngrok public URL"""
    try:
        tunnels = requests.get("http://localhost:4040/api/tunnels", timeout=2).json()
        for t in tunnels.get("tunnels", []):
            if t.get("proto") == "https":
                return t["public_url"]
    except Exception:
        return None


def print_setup_guide():
    ngrok_url = get_ngrok_url()

    print("\n" + "="*65)
    print("  EVA — Zoom Webhook Setup Guide")
    print("="*65)

    # Step 1
    print("\n📦 STEP 1 — Install & Start ngrok")
    print("─"*40)
    print("  1. Download ngrok from https://ngrok.com/download (free)")
    print("  2. Sign up for free account → get authtoken")
    print("  3. Run:  ngrok config add-authtoken YOUR_TOKEN")
    print("  4. Run:  ngrok http 8000")
    print()
    if ngrok_url:
        print(f"  ✅ ngrok is running!  URL: {ngrok_url}")
    else:
        print("  ❌ ngrok not detected. Start it first.")

    # Step 2
    webhook_url = f"{ngrok_url}/api/zoom/webhook" if ngrok_url else "https://YOUR_NGROK.ngrok-free.app/api/zoom/webhook"
    print("\n🔧 STEP 2 — Create Zoom App")
    print("─"*40)
    print("  1. Go to https://marketplace.zoom.us")
    print("  2. Click 'Develop' → 'Build App'")
    print("  3. Choose 'Webhook Only'")
    print("  4. App name: 'EVA Teaching Assistant'")
    print("  5. Click 'Continue' through all steps")

    # Step 3
    print("\n🔗 STEP 3 — Configure Event Subscriptions")
    print("─"*40)
    print("  1. In your Zoom app → 'Feature' tab")
    print("  2. Enable 'Event Subscriptions'")
    print("  3. Click '+ Add New Event Subscription'")
    print(f"  4. Endpoint URL:  {webhook_url}")
    print("  5. Click '+ Add Events' and select:")
    print("       ✅ Meeting → chat_message.sent")
    print("       ✅ Meeting → participant_joined")
    print("       ✅ Meeting → participant_left")
    print("  6. Click 'Save'")
    print("  7. Copy the 'Secret Token' shown")

    # Step 4
    print("\n🔐 STEP 4 — Add Secret Token to .env")
    print("─"*40)
    print("  Open your .env file and set:")
    print("  ZOOM_WEBHOOK_SECRET=paste_your_secret_token_here")

    # Step 5
    print("\n✅ STEP 5 — Validate URL (Zoom does this automatically)")
    print("─"*40)
    print("  After saving, Zoom sends a validation request to your URL.")
    print("  EVA handles this automatically.")
    print("  You should see: 'Webhook URL validated' in Zoom.")

    # Step 6
    print("\n🎓 STEP 6 — Use During Class")
    print("─"*40)
    print("  1. Start EVA session on teacher dashboard")
    print("  2. Start Zoom meeting as usual")
    print("  3. Students join Zoom normally — NO extra tab needed!")
    print("  4. When student types in Zoom chat:")
    print("       → EVA receives it via webhook automatically")
    print("       → Sentiment analyzed instantly")
    print("       → Teacher gets alert if confused/frustrated")

    print("\n" + "="*65)
    print("  Quick Check: http://localhost:8000/api/zoom/webhook-url")
    print("="*65 + "\n")


def test_webhook(secret: str = ""):
    """Send a test webhook to EVA to verify it's working"""
    print("\n🧪 Testing Zoom Webhook...")

    test_events = [
        {
            "name": "Student joins meeting",
            "payload": {
                "event": "meeting.participant_joined",
                "payload": {
                    "object": {
                        "participant": {
                            "user_name": "Test Student",
                            "user_id": "test123"
                        }
                    }
                }
            }
        },
        {
            "name": "Student sends confused message",
            "payload": {
                "event": "chat_message.sent",
                "payload": {
                    "object": {
                        "sender": {"display_name": "Test Student"},
                        "message": "I don't understand this topic at all, can you explain again?"
                    }
                }
            }
        },
        {
            "name": "Student asks for help",
            "payload": {
                "event": "chat_message.sent",
                "payload": {
                    "object": {
                        "sender": {"display_name": "Rahul Sharma"},
                        "message": "I need help with this question"
                    }
                }
            }
        },
        {
            "name": "Normal message (no alert expected)",
            "payload": {
                "event": "chat_message.sent",
                "payload": {
                    "object": {
                        "sender": {"display_name": "Priya Singh"},
                        "message": "Got it! Makes sense now"
                    }
                }
            }
        },
    ]

    for test in test_events:
        body = json.dumps(test["payload"])
        headers = {"Content-Type": "application/json"}

        # Add signature if secret provided
        if secret:
            ts = str(int(datetime.now().timestamp()))
            msg = f"v0:{ts}:{body}"
            sig = "v0=" + hmac.new(
                secret.encode(), msg.encode(), hashlib.sha256
            ).hexdigest()
            headers["x-zm-signature"]         = sig
            headers["x-zm-request-timestamp"] = ts

        try:
            r = requests.post(
                f"{EVA_SERVER}/api/zoom/webhook",
                data=body,
                headers=headers,
                timeout=5
            )
            status = "✅" if r.status_code == 200 else "❌"
            print(f"  {status} [{r.status_code}] {test['name']}")
            if r.status_code != 200:
                print(f"       Response: {r.text[:100]}")
        except Exception as e:
            print(f"  ❌ {test['name']}: {e}")

    print("\n  Check your teacher dashboard — alerts should appear!")
    print("  If no alerts: make sure a session is active first.\n")


def check_ngrok():
    """Check ngrok status and show webhook URL"""
    url = get_ngrok_url()
    if url:
        print(f"\n✅ ngrok is running!")
        print(f"   Public URL:      {url}")
        print(f"   Webhook endpoint: {url}/api/zoom/webhook")
        print(f"   Paste this URL in Zoom App → Event Subscriptions\n")
    else:
        print("\n❌ ngrok not running.")
        print("   Start it with:  ngrok http 8000\n")

    # Also check via EVA API
    try:
        r = requests.get(f"{EVA_SERVER}/api/zoom/webhook-url", timeout=3)
        data = r.json()
        if data.get("ngrok_url"):
            print(f"   EVA confirms: {data['webhook_endpoint']}")
    except Exception:
        print("   EVA server not reachable at localhost:8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVA Zoom Webhook Setup")
    parser.add_argument("--test",  action="store_true", help="Send test webhooks")
    parser.add_argument("--ngrok", action="store_true", help="Check ngrok URL")
    parser.add_argument("--secret", default="", help="Zoom webhook secret for signing")
    args = parser.parse_args()

    if args.ngrok:
        check_ngrok()
    elif args.test:
        test_webhook(args.secret)
    else:
        print_setup_guide()