#!/usr/bin/env python3
"""
Teams Meeting Detector - Detects Microsoft Teams meetings

Uses multiple detection methods:
1. Network connections: Teams uses UDP for audio/video during meetings
2. Window titles: Fallback for classic Teams (requires Screen Recording permission)

The network-based detection is more reliable for the new Teams app which
doesn't expose window titles.
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowListExcludeDesktopElements,
    kCGNullWindowID,
)


class MeetingState(Enum):
    """Possible meeting states."""
    NO_MEETING = "no_meeting"
    MEETING_ACTIVE = "meeting_active"


# Window title patterns indicating active call/meeting (for classic Teams)
MEETING_PATTERNS = [
    "Meeting with",
    "Call with",
    "| Chat",      # Chat window during call shows this
    "| Meeting",   # Meeting window
]

# Threshold for UDP connections to indicate active meeting
# During a meeting, Teams typically has 5+ UDP connections for audio/video
UDP_CONNECTION_THRESHOLD = 4


@dataclass
class TeamsDetectionResult:
    """Result of a Teams detection check."""
    is_meeting_active: bool
    meeting_title: Optional[str] = None
    teams_running: bool = False
    detection_method: str = "none"
    udp_connections: int = 0


def count_teams_network_connections() -> dict:
    """
    Count Teams network connections using lsof.

    Returns dict with 'tcp', 'udp', and 'total' counts.
    High UDP count (>4) indicates active audio/video call.
    """
    try:
        result = subprocess.run(
            ['lsof', '-i', '-n', '-P'],
            capture_output=True,
            text=True,
            timeout=5
        )

        tcp_count = 0
        udp_count = 0

        for line in result.stdout.split('\n'):
            # Match both "MSTeams" (new Teams) and "Microsoft Teams" (classic)
            if 'MSTeams' in line or 'Microsoft Teams' in line:
                if 'UDP' in line:
                    udp_count += 1
                elif 'TCP' in line and 'ESTABLISHED' in line:
                    tcp_count += 1

        return {'tcp': tcp_count, 'udp': udp_count, 'total': tcp_count + udp_count}

    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logging.getLogger(__name__).debug("Network check failed: %s", e)
        return {'tcp': 0, 'udp': 0, 'total': 0}


def is_teams_running() -> bool:
    """Check if Teams process is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-x', 'MSTeams'],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            return True

        # Also check for classic Teams
        result = subprocess.run(
            ['pgrep', '-f', 'Microsoft Teams'],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0

    except subprocess.SubprocessError:
        return False


def check_window_titles() -> Optional[str]:
    """
    Check Teams window titles for meeting indicators.

    Returns meeting title if found, None otherwise.
    Note: Requires Screen Recording permission on macOS 10.15+
    """
    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    window_list = CGWindowListCopyWindowInfo(options, kCGNullWindowID)

    if window_list is None:
        return None

    for window in window_list:
        owner_name = window.get("kCGWindowOwnerName", "") or ""
        window_title = window.get("kCGWindowName", "") or ""

        # Check if this is a Teams window with meeting pattern
        if "Microsoft Teams" in owner_name:
            for pattern in MEETING_PATTERNS:
                if pattern in window_title:
                    return window_title

    return None


def is_teams_call_active() -> TeamsDetectionResult:
    """
    Check if a Teams meeting/call is currently active.

    Uses two detection methods:
    1. Network connections (primary): UDP connections > threshold indicates meeting
    2. Window titles (fallback): For classic Teams with Screen Recording permission

    Returns TeamsDetectionResult with meeting status and detection details.
    """
    # Check if Teams is running at all
    teams_running = is_teams_running()

    if not teams_running:
        return TeamsDetectionResult(
            is_meeting_active=False,
            teams_running=False,
            detection_method="process_check"
        )

    # Method 1: Network-based detection (primary, works with new Teams)
    network_counts = count_teams_network_connections()
    udp_count = network_counts['udp']

    if udp_count >= UDP_CONNECTION_THRESHOLD:
        return TeamsDetectionResult(
            is_meeting_active=True,
            meeting_title="Teams Meeting",
            teams_running=True,
            detection_method="network",
            udp_connections=udp_count
        )

    # Method 2: Window title detection (fallback for classic Teams)
    meeting_title = check_window_titles()

    if meeting_title:
        return TeamsDetectionResult(
            is_meeting_active=True,
            meeting_title=meeting_title,
            teams_running=True,
            detection_method="window_title",
            udp_connections=udp_count
        )

    # No meeting detected
    return TeamsDetectionResult(
        is_meeting_active=False,
        meeting_title=None,
        teams_running=True,
        detection_method="none",
        udp_connections=udp_count
    )


def check_screen_recording_permission() -> bool:
    """
    Check if app likely has screen recording permission.

    Returns True if permission appears to be granted.
    Note: Network-based detection doesn't require this permission.
    """
    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    window_list = CGWindowListCopyWindowInfo(options, kCGNullWindowID)

    if window_list is None:
        return False

    # Check if we can see any window titles at all
    for window in window_list:
        title = window.get("kCGWindowName", "")
        if title:
            return True

    # No titles visible - either no windows or no permission
    return len(window_list) == 0


class TeamsDetector:
    """
    Background thread that monitors Teams meeting status.

    Polls at regular intervals and invokes callbacks on state changes.
    Implements grace period to handle brief disconnections.
    """

    def __init__(
        self,
        on_meeting_start: Callable[[str], None],
        on_meeting_end: Callable[[], None],
        poll_interval: float = 3.0,
        grace_period: float = 10.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Teams detector.

        Args:
            on_meeting_start: Callback when meeting starts, receives meeting title
            on_meeting_end: Callback when meeting ends
            poll_interval: Seconds between detection checks
            grace_period: Seconds to wait before declaring meeting ended
            logger: Optional logger instance
        """
        self.on_meeting_start = on_meeting_start
        self.on_meeting_end = on_meeting_end
        self.poll_interval = poll_interval
        self.grace_period = grace_period
        self.logger = logger or logging.getLogger(__name__)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_state = MeetingState.NO_MEETING
        self._meeting_ended_at: Optional[float] = None
        self._current_meeting_title: Optional[str] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the detection thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="TeamsDetector"
        )
        self._thread.start()
        self.logger.info("Teams detector started (poll=%.1fs, grace=%.1fs)",
                        self.poll_interval, self.grace_period)

    def stop(self):
        """Stop the detection thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.logger.info("Teams detector stopped")

    def _detection_loop(self):
        """Main detection loop running in background thread."""
        while self._running:
            try:
                self._check_meeting_status()
            except Exception as e:
                self.logger.error("Detection error: %s", e)

            time.sleep(self.poll_interval)

    def _check_meeting_status(self):
        """Check current meeting status and handle state transitions."""
        result = is_teams_call_active()

        with self._lock:
            if result.is_meeting_active:
                # Meeting is active - reset grace period timer
                self._meeting_ended_at = None

                if self._current_state == MeetingState.NO_MEETING:
                    # Transition: NO_MEETING -> MEETING_ACTIVE
                    self._current_state = MeetingState.MEETING_ACTIVE
                    self._current_meeting_title = result.meeting_title
                    self.logger.info(
                        "Meeting started: %s (detected via %s, UDP=%d)",
                        result.meeting_title,
                        result.detection_method,
                        result.udp_connections
                    )

                    # Invoke callback (on this background thread)
                    self._safe_callback(
                        self.on_meeting_start,
                        result.meeting_title or "Teams Meeting"
                    )
            else:
                # Meeting not detected
                if self._current_state == MeetingState.MEETING_ACTIVE:
                    # Start or continue grace period
                    if self._meeting_ended_at is None:
                        self._meeting_ended_at = time.time()
                        self.logger.debug(
                            "Meeting ended, starting grace period (%.1fs), UDP=%d",
                            self.grace_period,
                            result.udp_connections
                        )

                    # Check if grace period has elapsed
                    elapsed = time.time() - self._meeting_ended_at
                    if elapsed >= self.grace_period:
                        # Transition: MEETING_ACTIVE -> NO_MEETING
                        self._current_state = MeetingState.NO_MEETING
                        self._meeting_ended_at = None
                        self._current_meeting_title = None
                        self.logger.info("Meeting ended (grace period elapsed)")

                        self._safe_callback(self.on_meeting_end)

    def _safe_callback(self, callback: Callable, *args):
        """Safely invoke callback, catching any exceptions."""
        try:
            callback(*args)
        except Exception as e:
            self.logger.error("Callback error: %s", e)

    @property
    def is_meeting_active(self) -> bool:
        """Thread-safe check if meeting is currently active."""
        with self._lock:
            return self._current_state == MeetingState.MEETING_ACTIVE

    @property
    def current_meeting_title(self) -> Optional[str]:
        """Thread-safe access to current meeting title."""
        with self._lock:
            return self._current_meeting_title


if __name__ == "__main__":
    # Simple test - run detector and print status
    import sys

    logging.basicConfig(level=logging.DEBUG)

    def on_start(title):
        print(f"[MEETING START] {title}")

    def on_end():
        print("[MEETING END]")

    print("Testing Teams detection...")
    result = is_teams_call_active()
    print(f"  is_meeting_active: {result.is_meeting_active}")
    print(f"  teams_running: {result.teams_running}")
    print(f"  detection_method: {result.detection_method}")
    print(f"  udp_connections: {result.udp_connections}")
    print(f"  meeting_title: {result.meeting_title}")
    print()
    print(f"Screen recording permission: {check_screen_recording_permission()}")

    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        print("\nStarting continuous monitoring (Ctrl+C to stop)...")
        detector = TeamsDetector(
            on_meeting_start=on_start,
            on_meeting_end=on_end,
            poll_interval=2.0,
            grace_period=5.0
        )
        detector.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            detector.stop()
            print("\nStopped.")
