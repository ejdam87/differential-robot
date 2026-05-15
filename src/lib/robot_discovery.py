#!/usr/bin/env python3
"""
Listen for UDP broadcast packets containing JSON with "type": "intro"
and an "endpoint" field, for a limited time.

Returns a dict with robot info if found:
{
    "ip": "10.42.0.139",
    "port": 37020,
    "mac": "4c:d5:77:b7:36:7f",
}

This is a work of Martin Skalský (skalsky@mail.muni.cz)
"""

from __future__ import annotations

import json
import socket
import time
from typing import Optional


def discover_robot(
    target_mac: Optional[str] = None,
    listen_port: int = 37020,
    timeout: float = 10.0,
) -> Optional[dict]:
    """
    Listen for UDP broadcast "intro" packets for a limited time.

    Args:
        target_mac: Optional MAC address string; if given, only match this robot.
        listen_port: UDP port to listen on.
        timeout: How long to wait in seconds. Use 0 for infinite.

    Returns:
        dict with keys {ip, port, mac} or None if not found.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", listen_port))
    sock.settimeout(1.0)

    print(
        f"Listening for robot intro broadcasts on port {listen_port} for up to {timeout:.1f}s..."
    )
    if target_mac:
        print(f"Waiting specifically for MAC: {target_mac}")

    start_time = time.time()

    try:
        while timeout == 0 or time.time() - start_time < timeout:
            try:
                data, _addr = sock.recvfrom(65536)
                msg = json.loads(data.decode("utf-8"))
                msg_type = msg.get("type")
                endpoint = msg.get("endpoint")
                mac_address = msg.get("mac")

                if msg_type == "intro" and endpoint:
                    try:
                        ip_str, port_str = endpoint.split(":")
                        port_int = int(port_str)
                    except ValueError:
                        print(f"\nInvalid endpoint format: {endpoint}")
                        continue

                    result = {"ip": ip_str, "port": port_int, "mac": mac_address}

                    if target_mac:
                        if mac_address and mac_address.lower() == target_mac.lower():
                            print(f"\nOK: target robot found: {result}")
                            return result
                        print(f"\nDetected another robot ({mac_address}), ignoring...")
                        continue

                    print(f"\nOK: robot discovered: {result}")
                    return result

                else:
                    print(f"\nIgnoring packet: {msg}")

            except socket.timeout:
                print(".", end="", flush=True)
                continue
            except json.JSONDecodeError:
                print("\nReceived invalid JSON packet.")
                continue

        print("\nWarning: no matching robot broadcast received within timeout.")
        return None

    finally:
        sock.close()


# Example usage
if __name__ == "__main__":
    result = discover_robot(listen_port=37020, timeout=15.0, target_mac=None)
    if result:
        print(f"\nRobot found: {result}")
    else:
        print("\nNo robot found.")
