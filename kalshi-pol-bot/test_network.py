#!/usr/bin/env python3
"""
Network diagnostic - check if APIs are reachable
"""

import socket
import sys

apis_to_test = [
    ("api.kalshi.com", 443),
    ("demo-api.kalshi.co", 443),
    ("api.openai.com", 443),
    ("api.perplexity.ai", 443),
]

print("Testing network connectivity to APIs...\n")

all_good = True
for hostname, port in apis_to_test:
    try:
        # Try to resolve DNS
        ip = socket.gethostbyname(hostname)
        print(f"✅ {hostname:25s} → {ip}")
        
        # Try to connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((hostname, port))
        sock.close()
        
        if result == 0:
            print(f"   ✅ Port {port} is open")
        else:
            print(f"   ❌ Port {port} is blocked")
            all_good = False
            
    except socket.gaierror:
        print(f"❌ {hostname:25s} → DNS resolution failed!")
        all_good = False
    except Exception as e:
        print(f"❌ {hostname:25s} → Error: {e}")
        all_good = False
    
    print()

if all_good:
    print("✅ All APIs are reachable!")
    sys.exit(0)
else:
    print("❌ Some APIs are unreachable!")
    print("\nPossible fixes:")
    print("1. Check GitHub Codespace network settings")
    print("2. Try restarting the Codespace")
    print("3. Check if you're behind a firewall")
    print("4. Verify .env has correct API URLs")
    sys.exit(1)