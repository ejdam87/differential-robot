#!/usr/bin/env python3
"""UDP broadcaster script.

Sends a JSON message like:
{
    "type": "broadcast",
    "ip": "<your local IP>",
    "port": <some_port>
}
to the broadcast address of the current network (e.g. 192.168.1.255) on UDP port 37020.
"""

from __future__ import annotations

import argparse
import json
import socket


def get_local_ip() -> str:
    """Determine the primary local IP address (not loopback)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send packets; just determines the local interface used for this route.
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def calc_broadcast(ip: str) -> str:
    """Assume a /24 subnet and return the broadcast address."""
    parts = ip.split(".")
    if len(parts) == 4:
        parts[-1] = "255"
        return ".".join(parts)
    return "255.255.255.255"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send UDP broadcast JSON with current IP and port."
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port number to include in the JSON payload (default: 5000)",
    )
    parser.add_argument(
        "--broadcast-port",
        "-b",
        type=int,
        default=37020,
        help="UDP broadcast destination port (default: 37020)",
    )
    args = parser.parse_args()

    local_ip = get_local_ip()
    broadcast_ip = calc_broadcast(local_ip)

    message = {
        "type": "intro",
        "endpoint": f"{local_ip}:{args.port}",
    }

    data = json.dumps(message).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    dest = (broadcast_ip, args.broadcast_port)
    sock.sendto(data, dest)

    print(f"Broadcast sent to {dest}: {message}")

    sock.close()


if __name__ == "__main__":
    main()
