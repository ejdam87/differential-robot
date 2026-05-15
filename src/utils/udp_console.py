"""Simple UDP console for sending messages to a given IP and port."""

from __future__ import annotations

import argparse
import socket


def udp_console(ip: str, port: int) -> None:
    """Open a UDP console to send messages interactively to the given IP and port."""
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Connected to {ip}:{port} (UDP)")
        print("Type messages and press Enter to send. Type 'exit' to quit.\n")

        while True:
            text = input("> ")
            if text.lower() == "exit":
                print("Exiting.")
                break
            if not text.strip():
                continue
            sock.sendto(text.encode("utf-8"), (ip, port))

    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        if sock:
            sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple UDP console for sending messages to a given IP and port."
    )
    parser.add_argument("--ip", "-i", required=True, help="Target IP address")
    parser.add_argument(
        "--port", "-p", type=int, required=True, help="Target port number"
    )

    args = parser.parse_args()
    udp_console(args.ip, args.port)


if __name__ == "__main__":
    main()
