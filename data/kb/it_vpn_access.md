# VPN Access — IT Help

**Document ID:** it_vpn_access.md

## Overview
Johns Hopkins uses Cisco AnyConnect VPN to allow remote access to university systems, including library resources, internal tools, and some course materials.

## Installing VPN
1. Go to: `https://my.jh.edu` and log in with your JHED ID.
2. Navigate to **IT Downloads** > **Cisco AnyConnect VPN**.
3. Download and install the client for your operating system (Windows, macOS, Linux).
4. Launch Cisco AnyConnect and connect to: `vpn.jhu.edu`.

## Connecting to VPN
1. Open Cisco AnyConnect.
2. In the server field, enter: `vpn.jhu.edu`.
3. Log in with your **JHED ID** and password.
4. Complete DUO two-factor authentication.

## Common Issues
- **Connection refused**: Check that you're using `vpn.jhu.edu` (not a typo).
- **DUO not received**: Make sure your DUO device is enrolled at `https://my.jh.edu`. Re-enroll if needed.
- **Slow connection**: Try connecting to `vpn.jhu.edu/split` for split-tunnel mode (only university traffic goes through VPN).
- **macOS permissions**: On first use, macOS may block the VPN extension. Go to System Settings > Privacy & Security and allow the Cisco extension.

## Supported Platforms
Windows 10/11, macOS 12+, Ubuntu 20.04+.

## Getting Help
Contact the JHU IT Help Desk:
- Email: help@jhu.edu
- Phone: 410-516-HELP (4357)
- Live chat: `https://support.jhu.edu`
- Hours: Monday–Friday 8am–8pm ET, Saturday–Sunday 10am–6pm ET
