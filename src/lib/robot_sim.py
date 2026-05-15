#!/usr/bin/env python3
"""
Two-wheel robot UDP simulator compatible with RobotControl/robot_discovery.

Protocol expectations (from RobotControl):
- Client sends: {"command":"addClient","endpoint":"<ip>:<port>"}
- Simulator then periodically sends: {"type":"measurements", ...} to client's endpoint.
- Control: {"command":"control","leftPower":float,"rightPower":float}
- Engage/Disengage: {"command":"engage"} or {"command":"disengage"}

This simulator:
- Simulates differential drive kinematics.
- Reports encoder ticks for left/right wheels in measurement packets.
- Provides optional reflectance sensor readings from a grayscale map.


This is a work of Martin Skalský (skalsky@mail.muni.cz) 
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Optional, Tuple

from PIL import Image


def _now_us() -> int:
    return int(time.time() * 1_000_000)


class RobotSimulator:
    def __init__(
        self,
        robot_port: int = 37020,
        init_pose: Tuple[float, float, float] = (0, 0, 0),
        map_path: Optional[str] = None,
        pixel_size: float = 0.001,
        wheel_base_m: float = 0.083,
        max_velocity: float = 0.3,
        encoder_resolution: float = 0.001,
        meas_rate_hz: float = 20.0,
        sim_hz: float = 80.0,
        sensors_xy: Tuple[Tuple[float, float], ...] = (
            (0.02, 0.0395),
            (0.02, -0.0395),
            (-0.02, 0.0395),
            (-0.02, -0.0395),
        ),
    ) -> None:
        self.addr = ("127.0.0.1", robot_port)

        # Kinematic params
        self.L = wheel_base_m
        self.max_velocity = max_velocity
        self.tpm = 1.0 / encoder_resolution

        # State
        self.left_power = 0.0
        self.right_power = 0.0
        self.engaged = False

        self.x = init_pose[0]
        self.y = init_pose[1]
        self.yaw = init_pose[2]

        self.left_ticks = 0.0
        self.right_ticks = 0.0

        # Networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.settimeout(0.5)

        self.client_endpoint: Optional[Tuple[str, int]] = None

        # Threads
        self._run = False
        self._listener_th: Optional[threading.Thread] = None
        self._physics_th: Optional[threading.Thread] = None
        self._meas_th: Optional[threading.Thread] = None

        # Timing
        self.meas_period = 1.0 / max(1e-6, meas_rate_hz)
        self.dt_target = 1.0 / max(1e-6, sim_hz)

        # Concurrency
        self._state_lock = threading.Lock()

        # Map and sensing
        self.pixel_size = pixel_size
        self._load_map(map_path)
        self.sensors_xy = tuple(sensors_xy)

    # ---------------- Lifecycle ----------------

    def start(self) -> None:
        if self._run:
            return
        self._run = True
        self._listener_th = threading.Thread(target=self._listen_loop, daemon=True)
        self._physics_th = threading.Thread(target=self._physics_loop, daemon=True)
        self._meas_th = threading.Thread(target=self._measurements_loop, daemon=True)
        self._listener_th.start()
        self._physics_th.start()
        self._meas_th.start()
        print(f"RobotSimulator listening on UDP {self.addr[0]}:{self.addr[1]}")

    def stop(self) -> None:
        self._run = False
        try:
            self.sock.close()
        except Exception:
            pass
        for thread in (self._listener_th, self._physics_th, self._meas_th):
            if thread:
                thread.join(timeout=1.0)
        print("RobotSimulator stopped.")

    # ---------------- Loops ----------------

    def _listen_loop(self) -> None:
        while self._run:
            try:
                data, _addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception:
                continue

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            if not isinstance(msg, dict):
                continue

            cmd = msg.get("command")
            if cmd == "addClient":
                endpoint = msg.get("endpoint", "")
                ep_ip, ep_port = self._parse_endpoint(endpoint)
                if ep_ip is None:
                    continue

                with self._state_lock:
                    self.client_endpoint = (ep_ip, ep_port)

            elif cmd == "control":
                try:
                    lp = float(msg.get("leftPower", 0.0))
                    rp = float(msg.get("rightPower", 0.0))
                except Exception:
                    continue
                with self._state_lock:
                    self.left_power = max(-1.0, min(1.0, lp))
                    self.right_power = max(-1.0, min(1.0, rp))

            elif cmd == "engage":
                with self._state_lock:
                    self.engaged = True

            elif cmd == "disengage":
                with self._state_lock:
                    self.engaged = False

    def _physics_loop(self) -> None:
        # Integrate at ~80 Hz.
        last = time.time()
        while self._run:
            now = time.time()
            dt = now - last
            last = now

            with self._state_lock:
                lp = self.left_power if self.engaged else 0.0
                rp = self.right_power if self.engaged else 0.0

            v_l = lp * self.max_velocity
            v_r = rp * self.max_velocity

            # Differential drive kinematics.
            v = 0.5 * (v_r + v_l)
            omega = (v_l - v_r) / self.L

            # Integrate pose using exponential map (constant v, omega over dt).
            with self._state_lock:
                yaw0 = self.yaw
                if abs(omega) < 1e-9:
                    self.x += v * math.cos(yaw0) * dt
                    self.y += v * math.sin(yaw0) * dt
                    self.yaw = yaw0 + omega * dt
                else:
                    dtheta = omega * dt
                    self.x += (v / omega) * (math.sin(yaw0 + dtheta) - math.sin(yaw0))
                    self.y += (v / omega) * (-math.cos(yaw0 + dtheta) + math.cos(yaw0))
                    self.yaw = yaw0 + dtheta

                self.left_ticks += v_l * dt * self.tpm
                self.right_ticks += v_r * dt * self.tpm

            # Sleep to target rate.
            sleep_t = self.dt_target - (time.time() - now)
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                print(f"WARNING: physics loop behind by {sleep_t * 1000:.1f} ms")

    def _measurements_loop(self) -> None:
        next_t = time.time()
        while self._run:
            now = time.time()
            if now >= next_t:
                self._send_measurements()
                next_t += self.meas_period
            else:
                time.sleep(min(0.01, next_t - now))

    # ---------------- Helpers ----------------

    def _load_map(self, map_path: Optional[str]) -> None:
        """Load grayscale map if provided. If not available, disable map sensing."""
        self.map_img = None
        self._map_px = None
        self.map_w = 0
        self.map_h = 0
        if not map_path:
            return

        img = Image.open(map_path).convert("L")
        self.map_img = img
        self.map_w, self.map_h = img.size
        self._map_px = img.load()

    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world meters (x, y) to image pixel (u, v).

        World frame: (0,0) at top-left of map, +x right, +y down.
        """
        u = int(round(x / self.pixel_size))
        v = int(round(y / self.pixel_size))
        return u, v

    def _sensor_world_pos(
        self, robot_x: float, robot_y: float, robot_yaw: float, sx: float, sy: float
    ) -> Tuple[float, float]:
        """
        Sensor position in world given robot pose and sensor offset (sx, sy) in robot frame.

        Robot frame: x forward, y left.
        """
        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)
        wx = robot_x + c * sx - s * sy
        wy = robot_y + s * sx + c * sy
        return wx, wy

    def _read_pixel(self, wx: float, wy: float) -> float:
        """
        Return reflectance value where 0 is black and 1020 is white.
        Out-of-bounds returns 0 (black).
        """
        if self.map_img is None:
            return 0

        u, v = self._world_to_pixel(wx, wy)
        if 0 <= u < self.map_w and 0 <= v < self.map_h:
            return self._map_px[u, v] * 4
        return 0

    def get_pose(self) -> Tuple[float, float, float]:
        """
        Return the current robot pose as (x, y, yaw).

        Units: meters for x/y and radians for yaw.
        """
        with self._state_lock:
            return self.x, self.y, self.yaw

    def get_sensor_poses(self) -> list[Tuple[float, float]]:
        """
        Compute poses of all sensors in the global frame based on current robot pose.

        Returns:
            List of (x, y) positions.
        """
        with self._state_lock:
            robot_x, robot_y, robot_yaw = self.x, self.y, self.yaw
        sensor_poses = []
        for sx, sy in self.sensors_xy:
            wx, wy = self._sensor_world_pos(robot_x, robot_y, robot_yaw, sx, sy)
            sensor_poses.append((wx, wy))
        return sensor_poses

    def _parse_endpoint(self, endpoint: str) -> Tuple[Optional[str], Optional[int]]:
        try:
            ip_str, port_str = endpoint.strip().split(":")
            return ip_str, int(port_str)
        except Exception:
            return None, None

    def _send_measurements(self) -> None:
        if not self.client_endpoint:
            return
        try:
            with self._state_lock:
                client_ep = self.client_endpoint

                readings = []
                for sx, sy in self.sensors_xy:
                    wx, wy = self._sensor_world_pos(self.x, self.y, self.yaw, sx, sy)
                    readings.append(self._read_pixel(wx, wy))

                msg = {
                    "type": "measurements",
                    "timestamp": _now_us(),
                    "leftEnc": int(self.left_ticks),
                    "rightEnc": int(self.right_ticks),
                    "reflect": readings,
                    "pose": {"x": self.x, "y": self.y, "yaw": self.yaw},
                }
            data = json.dumps(msg).encode("utf-8")
            self.sock.sendto(data, client_ep)
        except Exception:
            pass


if __name__ == "__main__":
    sim = RobotSimulator(bind_ip="127.0.0.1", robot_port=50000)
    try:
        sim.start()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
