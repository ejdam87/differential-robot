"""
Windowless keyboard listener using pynput.

This is a work of Martin Skalský (skalsky@mail.muni.cz) 
"""

from __future__ import annotations

from typing import Optional, Set, Tuple
import threading

from pynput import keyboard

SPECIAL_KEYS = {
    keyboard.Key.space: " ",
    keyboard.Key.up: "up",
    keyboard.Key.down: "down",
    keyboard.Key.left: "left",
    keyboard.Key.right: "right",
    keyboard.Key.esc: "esc",
    keyboard.Key.enter: "enter",
    keyboard.Key.tab: "tab",
    keyboard.Key.shift: "shift",
    keyboard.Key.ctrl: "ctrl",
    keyboard.Key.alt: "alt",
    keyboard.Key.cmd: "cmd",
    keyboard.Key.backspace: "backspace",
    keyboard.Key.delete: "delete",
}


class KeyboardMonitor:
    """
    Keyboard listener (no window) using pynput.

    API:
    - start(), stop()
    - get_pressed()  -> set of keys newly pressed since last call (then cleared)
    - get_released() -> set of keys newly released since last call (then cleared)
    - get_hold()     -> set of keys currently held (not cleared)
    - get_keys()     -> (pressed, released, hold)
    """

    def __init__(self) -> None:
        self._pressed_since: Set[str] = set()
        self._released_since: Set[str] = set()
        self._hold: Set[str] = set()

        self._lock = threading.Lock()
        self._listener: Optional[keyboard.Listener] = None

    def _normalize_key(self, key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
        """Return a normalized string for keys we care about; None otherwise."""
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        return SPECIAL_KEYS.get(key)

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        normalized = self._normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            # Only mark as newly pressed if it wasn't already held.
            if normalized not in self._hold:
                self._pressed_since.add(normalized)
            self._hold.add(normalized)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        normalized = self._normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            if normalized in self._hold:
                self._hold.remove(normalized)
                self._released_since.add(normalized)

    def start(self) -> None:
        """Start the keyboard listener."""
        if self._listener is None:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()

    def stop(self) -> None:
        """Stop the keyboard listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_pressed(self) -> Set[str]:
        """Return newly pressed keys since last call and clear the buffer."""
        with self._lock:
            out = set(self._pressed_since)
            self._pressed_since.clear()
            return out

    def get_released(self) -> Set[str]:
        """Return newly released keys since last call and clear the buffer."""
        with self._lock:
            out = set(self._released_since)
            self._released_since.clear()
            return out

    def get_hold(self) -> Set[str]:
        """Return currently held keys (not cleared)."""
        with self._lock:
            return set(self._hold)

    def get_keys(self, clear: bool = True) -> Tuple[Set[str], Set[str], Set[str]]:
        """Return (pressed, released, hold)."""
        with self._lock:
            pressed = set(self._pressed_since)
            released = set(self._released_since)
            hold = set(self._hold)
            if clear:
                self._pressed_since.clear()
                self._released_since.clear()
            return pressed, released, hold


if __name__ == "__main__":
    import time

    monitor = KeyboardMonitor()
    monitor.start()
    try:
        while True:
            pressed, released, held = monitor.get_keys()
            if pressed or released:
                print("pressed:", pressed, "released:", released, "held:", held)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
