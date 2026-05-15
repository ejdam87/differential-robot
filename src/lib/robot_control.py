#!/usr/bin/env python3
"""
RobotControl: simple UDP JSON communication interface.

Features:
- Connects to a remote robot via UDP.
- Initiates handshake with command "addClient".
- Waits for responses with type "measurements".
- Retries handshake until success.
- Receives and parses measurement JSON messages.
- Provides methods to:
  - get_measurements() -> list of new measurement dicts
  - set_control(left_motor, right_motor)
  - engage(enable=True) / disengage()

This is a work of Martin Skalský (skalsky@mail.muni.cz)
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class RobotControl:
    def __init__(
        self,
        robot_ip: str,
        robot_port: int = 37020,
        local_port: Optional[int] = None,
        retry_interval: float = 2.0,
    ) -> None:
        """
        Args:
            robot_ip: IP address of the robot to communicate with.
            robot_port: UDP port on the robot to send messages to.
            local_port: Local UDP port to listen for responses; ephemeral if None.
            retry_interval: Seconds between retry attempts.
        """
        self.robot_addr: Tuple[str, int] = (robot_ip, robot_port)
        self.retry_interval = float(retry_interval)
        self.running = True
        self._lock = threading.Lock()
        self.measurements: List[Dict[str, Any]] = []
        self.sock: Optional[socket.socket] = None
        self.listener_thread: Optional[threading.Thread] = None
        self.handshake_done = False

        # Determine local IP
        self.local_ip = self._get_local_ip()

        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if local_port is None:
            self.sock.bind(("", 0))
        else:
            self.sock.bind(("", local_port))

        self.sock.settimeout(1.0)  # used for handshake waiting

        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

        # Try to establish communication
        self._initiate_handshake()

    def _get_local_ip(self) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((self.robot_addr[0], 80))
            ip = sock.getsockname()[0]
        except Exception:
            print("Warning: unable to determine local IP address.")
            ip = "127.0.0.1"
        finally:
            sock.close()
        return ip

    def _initiate_handshake(self) -> None:
        """Repeatedly send addClient command until a measurements packet arrives."""
        payload = {
            "command": "addClient",
            "endpoint": f"{self.local_ip}:{self.sock.getsockname()[1]}",
        }
        data = json.dumps(payload).encode("utf-8")

        print(
            "Attempting to register with robot at "
            f"{self.robot_addr} from {self.sock.getsockname()[0]}:{self.sock.getsockname()[1]} ..."
        )
        while not self.handshake_done and self.running:
            try:
                self.sock.sendto(data, self.robot_addr)
                print(f"Sent addClient -> {payload}")
                # Wait briefly for measurement response
                start_time = time.time()
                while time.time() - start_time < self.retry_interval:
                    if self.handshake_done:
                        break
                    time.sleep(0.1)
                if not self.handshake_done:
                    print("No measurements received yet - retrying...")
            except Exception as exc:
                print(f"Error sending addClient: {exc}")
                time.sleep(self.retry_interval)

        if self.handshake_done:
            print("Handshake successful - robot is sending measurements.")

    def _listen_loop(self) -> None:
        """Background listener for incoming UDP messages."""
        while self.running:
            try:
                data, _addr = self.sock.recvfrom(65536)
                msg = json.loads(data.decode("utf-8"))
                if isinstance(msg, dict):
                    if msg.get("type") == "measurements":
                        self.handshake_done = True
                        self._handle_measurements(msg)
                    else:
                        print(f"Non-measurement message: {msg}")
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                print(f"Cannot decode message: {data}")
                continue
            except OSError:
                break  # socket closed
            except Exception as exc:
                print(f"Listener error: {exc}")
                continue

    def _handle_measurements(self, msg: Dict[str, Any]) -> None:
        """Append new measurement packet to list."""
        with self._lock:
            self.measurements.append(msg)

    def get_measurements(self) -> List[Dict[str, Any]]:
        """Return all collected measurement messages and clear the buffer."""
        with self._lock:
            current = list(self.measurements)
            self.measurements.clear()
        return current

    def getMeasurements(self) -> List[Dict[str, Any]]:
        """Backward-compatible alias for get_measurements()."""
        return self.get_measurements()

    def set_control(self, left_motor: float, right_motor: float) -> None:
        """Send motor control command to robot."""
        payload = {
            "command": "control",
            "leftPower": float(left_motor),
            "rightPower": float(right_motor),
        }
        self._send_json(payload)

    def setControl(self, left_motor: float, right_motor: float) -> None:
        """Backward-compatible alias for set_control()."""
        self.set_control(left_motor, right_motor)

    def engage(self, enable: bool = True) -> None:
        """Send engage command to robot."""
        payload = {"command": "engage" if enable else "disengage"}
        self._send_json(payload)

    def disengage(self) -> None:
        """Explicit disengage command."""
        payload = {"command": "disengage"}
        self._send_json(payload)

    def _send_json(self, payload: Dict[str, Any]) -> None:
        """Helper: send JSON payload to robot."""
        try:
            data = json.dumps(payload).encode("utf-8")
            self.sock.sendto(data, self.robot_addr)
        except Exception as exc:
            print(f"Send error: {exc}")

    def close(self) -> None:
        """Gracefully stop listener and close socket."""
        self.running = False
        if self.sock:
            self.sock.close()
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)


# Example usage
if __name__ == "__main__":
    robot = RobotControl(robot_ip="192.168.1.100", robot_port=50000)
    try:
        while True:
            for measurement in robot.get_measurements():
                print("Measurement:", measurement)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        robot.close()
