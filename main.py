```python
#!/usr/bin/env python3
"""
Log File Anomaly Detector
A CLI tool for detecting anomalies in cloud/DevOps log files.

Supports multiple log formats, sliding-window statistics,
pattern rarity scoring, and multi-method anomaly detection.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """Normalized log entry."""
    timestamp: datetime | None
    level: str                  # DEBUG, INFO, WARN, ERROR, CRITICAL, or UNKNOWN
    message: str
    source_file: str = ""
    raw_line: str = ""


@dataclass
class AnomalyScore:
    """Score for a single anomaly window."""
    window_start: str
    window_end: str
    metric: str
    observed: float
    expected: float
    z_score: float
    severity: str              # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class AnomalyReport:
    """Complete report for a single detected anomaly."""
    timestamp: str
    type: str                   # frequency_spike, error_rate, rare_pattern, ip_anomaly
    description: str
    severity: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOG_LEVELS = {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "WARN": 30,
              "INFO": 20, "DEBUG": 10, "TRACE": 5}

TIMESTAMP_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ISO-8601 / RFC3339 variants
    ("iso", re.compile(
        r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)")),
    # Common log format: [10/Oct/2000:13:55:36 -0700]
    ("clf", re.compile(
        r"\[(\d{1,2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]?\d{4})\]")),
    # Syslog: Mon DD HH:MM:SS
    ("syslog", re.compile(
        r"^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")),
    # Nginx/Apache combined without brackets
    ("nginx_acc", re.compile(
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})")),
    # epoch ms
    ("epoch", re.compile(r"^\s*(\d{10,13})")),
]

TIMESTAMP_PARSERS: list[tuple[str, str, re.Pattern]] = [
    ("iso",      "%Y-%m-%dT%H:%M:%S%z"),
    ("iso_nz",   "%Y-%m-%dT%H:%M:%S"),
    ("clf",      "%d/%b/%Y:%H:%M:%S %z"),
    ("syslog",   "%b %d %H:%M:%S"),
    ("nginx_acc","%Y-%m-%d %H:%M:%S"),
]

SYSLOG_MONTHS = {m: i+1 for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}

IP_PATTERN = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
                        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")

HTTP_STATUS_PATTERN = re.compile(r"\b([1-5]\d{2})\b")

ERROR_KEYWORDS = re.compile(
    r"\b(error|exception|fail|fatal|panic|crash|unavailable|"
    r"timeout|refused|denied|OOM|killed|SIGTERM|traceback)\b",
    re.IGNORECASE,
)


def _parse_syslog_ts(match_str: str, fallback_year: int) -> datetime:
    """Parse a syslog-style timestamp (no year) into a datetime."""
    dt = datetime.strptime(match_str, "%b %d %H:%M:%S")
    return dt.replace(year=fallback_year)


def parse_timestamp(raw: str) -> datetime | None:
    """Try all known timestamp formats; return the first successful parse."""
    if not raw:
        return None
    raw_s = raw.strip()

    # Epoch milliseconds
    if raw_s.isdigit():
        n = int(raw_s)
        if n > 1e12:       # milliseconds
            n //= 1000
        try:
            return datetime.fromtimestamp(n, tz=None)
        except OSError:
            return None

    for label, pat in TIMESTAMP_PATTERNS:
        m = pat.search(raw_s)
        if not m:
            continue
        ts_str = m.group(1)

        # Special: syslog needs a fallback year
        if label == "syslog":
            try:
                return _parse_syslog_ts(ts_str, datetime.now().year)
            except ValueError:
                continue

        for parser_fmt in TIMESTAMP_PARSERS:
            fmt = parser_fmt[1]
            try:
                dt = datetime.strptime(ts_str, fmt)
                return dt
            except ValueError:
                continue

    return None


def _detect_level(line: str, match: re.Match | None = None) -> str:
    """Extract log level from a line."""
    # Direct token match
    upper = line.upper()
    for kw, score in LOG_LEVELS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", upper):
            return kw
    # HTTP status codes
    sm = HTTP_STATUS_PATTERN.search(line)
    if sm:
        code = int(sm.group(1))
        if code >= 500:
            return "ERROR"
        if code >= 400:
            return "WARN"
        if code >= 300:
            return "INFO"
    if ERROR_KEYWORDS.search(line):
        return "ERROR"
    return "UNKNOWN"


def _normalise_message(line: str, ts_len: int = 0) -> str:
    """Strip timestamp and noise to get the core message."""
    msg = line.strip()
    # Remove leading bracketed timestamp
    msg = re.sub(r"^\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]\s*", "", msg)
    msg = re.sub(r"^\[\d{1,2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]?\d{4}\]\s*", "", msg)
    # Deduplicate repeated whitespace
    msg = re.sub(r"\s{2,}", " ", msg).strip()
    return msg


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class LogParser:
    """
    Parses one or more log files into normalized LogEntry objects.
    Auto-detects per-file format hints from the first 20 lines.
    """

    JSON_RE = re.compile(r"^\s*\{.*\}\s*$")
    KEY_VALUE_RE = re.compile(r"(\w+)=(?:\"([^\"]*)\"|([\w.:/\-]+))")

    def __init__(self, input_paths: Iterable[str], fallback_year: int | None = None):
        self.input_paths = [str(p) for p in input_paths]
        self._fallback_year = fallback_year or datetime.now().year

    # -- public ----------------------------------------------------------

    def parse_all(self) -> list[LogEntry]:
        entries: list[LogEntry] = []
        for path in self.input_paths:
            entries.extend(self._parse_file(path))
        entries.sort(key=lambda e: e.timestamp or datetime.min)
        return entries

    # -- internals -------------------------------------------------------

    def _parse_file(self, path: str) -> list[LogEntry]:
        entries: list[LogEntry] = []
        p = Path(path)
        if not p.exists():
            logging.error("Input file not found: %s", path)
            return entries

        # Peek at first few lines to guess format
        peek = self._peek_lines(p, n=20)
        fmt = self._guess_format(peek)
        logging.info("Detected format '%s' for %s (%d lines peeked)",
                     fmt, path, len(peek))

        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.rstrip("\n\r")
                if not raw_line.strip():
                    continue
                entry = self._parse_line(raw_line, fmt, path, lineno)
                if entry is not None:
                    entries.append(entry)
        logging.debug("Parsed %d entries from %s", len(entries), path)
        return entries

    def _peek_lines(self, path: Path, n: int) -> list[str]:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return [next(fh, "").rstrip() for _ in range(n)]

    def _guess_format(self, lines: list[str]) -> str:
        json_count = sum(1 for l in lines if self.JSON_RE.match(l))
        kv_count = sum(1 for l in lines if self.KEY_VALUE_RE.search(l))
        has_ts = sum(1 for l in lines if parse_timestamp(l) is not None)
        total = max(len(lines), 1)
        if json_count / total > 0.3:
            return "json"
        if kv_count / total > 0.3:
            return "keyvalue"
        if has_ts / total > 0.3:
            return "generic"
        return "generic"

    def _parse_line(self, line: str, fmt: str, src: str, lineno: int) -> LogEntry | None:
        ts: datetime | None = None
        message = line
        level = "UNKNOWN"

        if fmt == "json":
            return self._parse_json(line, src)
        if fmt == "keyvalue":
            return self._parse_kv(line, src)

        # Generic: extract timestamp then rest is message
        ts = parse_timestamp(line)
        message = _normalise_message(line) if ts else line
        level = _detect_level(line)
        return LogEntry(timestamp=ts, level=level, message=message,
                        source_file=src, raw_line=line)

    def _parse_json(self, line: str, src: str) -> LogEntry | None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Not JSON after all – fall through to generic
            return self._parse_line(line, "generic", src, 0)
        if not isinstance(obj, dict):
            return self._parse_line(line, "generic", src, 0)

        ts = None
        for key in ("time", "timestamp", "@timestamp", "ts", "datetime"):
            if key in obj:
                ts = parse_timestamp(str(obj[key]))
                if ts:
                    break

        # Extract level
        level = "UNKNOWN"
        for key in ("level", "severity", "loglevel", "log_level"):
            if key in obj:
                level = _detect_level(str(obj[key]))
                break

        # Extract message
        message_parts: list[str] = []
        for key in ("msg", "message", "log"):
            if key in obj:
                message_parts.append(str(obj[key]))
        message = " ".join(message_parts) if message_parts else json.dumps(obj, default=str)

        return LogEntry(timestamp=ts, level=level, message=message,
                        source_file=src, raw_line=line)

    def _parse_kv(self, line: str, src: str) -> LogEntry | None:
        ts = parse_timestamp(line)
        kvs = dict(self.KEY_VALUE_RE.findall(line))
        level_str = kvs.get("level", kvs.get("severity", kvs.get("loglevel", "")))
        message = kvs.get("msg", kvs.get("message", line))
        level = _detect_level(level_str or message)
        return LogEntry(timestamp=ts, level=level, message=message,
                        source_file=src, raw_line=line)


# ---------------------------------------------------------------------------
# Anomaly Detectors
# ---------------------------------------------------------------------------

class WindowCalculator:
    """Split a sorted sequence of timestamps into fixed-size windows."""

    def __init__(self, entries: list[LogEntry], window_seconds: int = 60):
        self.entries = [e for e in entries if e.timestamp is not None]
        self.window_td = timedelta(seconds=window_seconds)
        if not self.entries:
            self.windows: list[tuple[datetime, datetime, list[LogEntry]]] = []
            return

        t0 = self.entries[0].timestamp
        # Align to start
        end = t0 + self.window_td
        self.windows = []
        bucket: list[LogEntry] = []
        for entry in self.entries:
            if entry.timestamp < end:
                bucket.append(entry)
            else:
                if bucket:
                    self.windows.append((t0, end, bucket))
                # Advance window so there's no gap
                while end <= entry.timestamp:
                    end += self.window_td
                t0 = end - self.window_td
                bucket = [entry]
        if bucket:
            self.windows.append((t0, end, bucket))

    @property
    def count(self) -> int:
        return len(self.windows)

    def entries_per_window(self) -> list[int]:
        return [len(w[2]) for w in self.windows]

    def error_count_per_window(self) -> list[int]:
        return [sum(1 for e in w[2] if e.level in ("ERROR", "CRITICAL"))
                for w in self.windows]

    def level_counts_per_window(self) -> dict[str, list[int]]:
        buckets: dict[str, list[int]] = defaultdict(list)
        for w in self.windows:
            cnt = Counter(e.level for e in w[2])
            for lvl, c in cnt.items():
                buckets[lvl].append(c)
        return dict(buckets)


class ZScoreStatistics:
    """Compute mean, std, z-score for a numeric series."""

    def __init__(self, values: Iterable[float]):
        self.values = list(values)
        self.n = len(self.values)
        if self.n == 0:
            self.mean = 0.0
            self.std = 0.0
            return
        self.mean = sum(self.values) / self.n
        variance = sum((v - self.mean) ** 2 for v in self.values) / max(self.n - 1, 1)
        self.std = variance ** 0.5

    def z_score(self, value: float) -> float:
        if self.std == 0:
            return 0.0
        return (value - self.mean) / self.std

    def percentile(self, pct: float) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        k = (pct / 100.0) * (len(s) - 1)
        f = int(k)
        c = f + 1 if f + 1 < len(s) else f
        d = k - f
        return s[f] + d * (s[c] - s[f])


def _severity_from_z(z: float, threshold_high: float = 2.0,
                     threshold_crit: float = 3.5) -> str:
    az = abs(z)
    if az >= threshold_crit:
        return "CRITICAL"
    if az >= threshold_high:
        return "HIGH"
    if az >= threshold_high * 0.6:
        return "MEDIUM"
    return "LOW"


class AnomalyDetector:
    """
    Multi-strategy anomaly detector operating on parsed LogEntry sequences.

    Strategies:
      - frequency:     sliding-window count z-score
      - error_rate:    error-count z-score against all-entry mean
      - pattern_rarity: rare message-template scoring via hashing
      - ip_anomaly:    unusual IPs / endpoints (access-log oriented)
    """

    def __init__(self, entries: list[LogEntry],
                 window_seconds: int = 60,
                 z_threshold: float = 2.0,
                 min_windows: int = 5):
        self.entries = entries
        self.window_seconds = window_seconds
        self.z_threshold = z_threshold
        self.min_windows = min_windows
        self.wc: WindowCalculator | None = None
        self._report: list[AnomalyReport] = []

    # -- public ----------------------------------------------------------

    def detect(self, strategies: list[str] | None = None) -> list[AnomalyReport]:
        if strategies is None:
            strategies = ["frequency", "error_rate", "pattern_rarity", "ip_anomaly"]
        self._report = []
        for strat in strategies:
            handler = getattr(self, f"_detect_{strat}", None)
            if handler is None:
                logging.warning("Unknown strategy: %s", strat)
                continue
            handler()
        self._report.sort(key=lambda r: r.timestamp)
        return self._report

    # -- strategy implementations ----------------------------------------

    def _detect_frequency(self) -> None:
        if not self.entries:
            return
        wc = WindowCalculator(self.entries, self.window_seconds)
        if wc.count < self.min_windows:
            logging.debug("frequency: only %d windows, need >= %d", wc.count, self.min_windows)
            return
        vals = [float(c) for c in wc.entries_per_window()]
        stats = ZScoreStatistics(vals)
        for (ws, we), bucket in zip(
                [(w[0], w[1]) for w in wc.windows],
                wc.windows):
            observed = len(bucket)
            z = stats.z_score(float(observed))
            if abs(z) >= self.z_threshold:
                self._report.append(AnomalyReport(
                    timestamp=ws.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    type="frequency",
                    description=(
                        f"Unusual event frequency: {observed} events in "
                        f"{self.window_seconds}s window (expected ~{stats.mean:.1f}, "
                        f"z={z:.2f})."
                    ),
                    severity=_severity_from_z(z),
                    details={"observed": observed, "expected_mean": round(stats.mean, 2),
                             "std": round(stats.std, 2), "z_score": round(z, 2),
                             "window_end": we.strftime("%Y-%m-%dT%H:%M:%SZ")},
                ))

    def _detect_error_rate(self) -> None:
        if not self.entries:
            return
        wc = WindowCalculator(self.entries, self.window_seconds)
        if wc.count < self.min_windows:
            return
        err_vals = [float(c) for c in wc.error_count_per_window()]
        all_vals = [float(c) for c in wc.entries_per_window()]
        err_stats = ZScoreStatistics(err_vals)
        total_stats = ZScoreStatistics(all_vals)

        for (ws, we), bucket in zip(
                [(w[0], w[1]) for w in wc.windows],
                wc.windows):
            n_err = sum(1 for e in bucket if e.level in ("ERROR", "CRITICAL"))
            n_total = len(bucket)
            rate = n_err / n_total if n_total else 0.0
            z = err_stats.z_score(float(n_err))
            if abs(z) >= self.z_threshold:
                self._report.append(AnomalyReport(
                    timestamp=ws.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    type="error_rate",
                    description=(
                        f"Error rate spike: {n_err} errors out of {n_total} events "
                        f"(rate={rate:.1%}, z={z:.2f})."
                    ),
                    severity=_severity_from_z(z),
                    details={"errors": n_err, "total": n_total,
                             "error_rate": round(rate, 4),
                             "expected_errors": round(err_stats.mean, 2),
                             "z_score": round(z, 2),
                             "window_end": we.strftime("%Y-%m-%dT%H:%M:%SZ")},
                ))

    def _detect_pattern_rarity(self) -> None:
        """
        Hash each message into a template fingerprint; flag windows where
        previously unseen or ultra-rare fingerprints appear.
        """
        if not self.entries or len(self.entries) < 10:
            return

        # Build global fingerprint counter
        fp_counter: Counter[str] = Counter()
        for e in self.entries:
            fp = self._fingerprint(e.message)
            fp_counter[fp] += 1

        rare_threshold = max(3, len(self.entries) * 0.001)
        top_fp = fp_counter.most_common(20)
        frequent_fps = {fp for fp, _ in top_fp}

        wc = WindowCalculator(self.entries, self.window_seconds)
        if wc.count < 2:
            return

        for (ws, we), bucket in zip(
                [(w[0], w[1]) for w in wc.windows],
                wc.windows):
            rare_msgs: list[str] = []
            for e in bucket:
                fp = self._fingerprint(e.message)
                if fp_counter[fp] <= rare_threshold and fp not in frequent_fps:
                    rare_msgs.append(e.message)
            if len(rare_msgs) >= 3:
                self._report.append(AnomalyReport(
                    timestamp=ws.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    type="rare_pattern",
                    description=(
                        f"Window contains {len(rare_msgs)} unusually rare log messages "
                        f"(below threshold {rare_threshold}). First: "
                        f"{rare_msgs[0][:120]}."
                    ),
                    severity=_severity_from_z(len(rare_msgs), threshold_high=1.5),
                    details={"rare_count": len(rare_msgs),
                             "threshold": rare_threshold,
                             "sample_messages": rare_msgs[:3]},
                ))

    def _detect_ip_anomaly(self) -> None:
        """Detect suspicious IPs: very high request counts or new rarely-seen IPs."""
        ip_counter: Counter[str] = Counter()
        ip_first_seen: dict[str, datetime] = {}
        ip_entries: dict[str, list[LogEntry]] = defaultdict(list)

        for e in self.entries:
            ips = IP_PATTERN.findall(e.raw_line)
            for ip in ips:
                ip_counter[ip] += 1
                ip_entries[ip].append(e)
                if ip not in ip_first_seen:
                    ip_first_seen[ip] = e.timestamp or datetime.min

        if not ip_counter or len(ip_counter) < 3:
            return

        vals = [float(c) for c in ip_counter.values()]
        stats = ZScoreStatistics(vals)
        top_ips = {ip for ip, _ in ip_counter.most_common(5)}

        for ip, count in ip_counter.items():
            z = stats.z_score(float(count))
            if abs(z) >= self.z_threshold and ip not in top_ips:
                ts_samples = sorted(
                    (e.timestamp for e in ip_entries[ip] if e.timestamp),
                )
                self._report.append(AnomalyReport(
                    timestamp=(ts_samples[0].strftime("%Y-%m-%dT%H:%M:%SZ")
                               if ts_samples else "unknown"),
                    type="ip_anomaly",
                    description=(
                        f"IP {ip} exhibited anomalous activity: "
                        f"{count} occurrences (z={z:.2f}), "
                        f"first seen {ts_samples[0].isoformat() if ts_samples else '?'}."
                    ),
                    severity=_severity_from_z(z),
                    details={"ip": ip, "count": count,
                             "z_score": round(z, 2),
                             "first_seen": ts_samples[0].isoformat()
                                           if ts_samples else None},
                ))

    # -- internals -------------------------------------------------------

    @staticmethod
    def _fingerprint(message: str) -> str:
        """
        Convert a message to a stable fingerprint by replacing numbers,
        IPs, UUIDs and hex tokens with placeholders.
        """
        s = message
        s = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b", "<TS>", s)
        s = re.sub(IP_PATTERN, "<IP>", s)
        s = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>", s)
        s = re.sub(r"\b\d+\b", "<N>", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        import hashlib
        return hashlib.md5(s.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_text(report: list[AnomalyReport], entries: list[LogEntry]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  LOG FILE ANOMALY DETECTOR — REPORT")
    lines.append("=" * 72)
    lines.append("")

    # Summary
    total = len(entries)
    ts_range = ""
    if entries:
        timestamps = [e.timestamp for e in entries if e.timestamp]
        if timestamps:
            ts_range = (f"{min(timestamps).isoformat()}  →  {max(timestamps).isoformat()}")
    levels = Counter(e.level for e in entries)
    lines.append(f"  Entries analysed : {total}")
    lines.append(f"  Time range       : {ts_range or 'N/A'}")
    lines.append(f"  Anomalies found  : {len(report)}")
    lines.append("")
    lines.append("  Level distribution:")
    for lvl in ("CRITICAL", "ERROR", "WARN", "INFO", "DEBUG", "UNKNOWN"):
        c = levels.get(lvl, 0)
        if c:
            pct = c / total * 100 if total else 0
            bar = "█" * max(1, round(pct / 2))
            lines.append(f"    {lvl:<8} {c:>6}  ({pct:5.1f}%)  {bar}")
    lines.append("")

    if not report:
        lines.append("  ✓ No anomalies detected.")
        lines.append("")
        return "\n".join(lines)

    lines.append("-" * 72)
    lines.append("  ANOMALIES")
    lines.append("-" * 72)

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    report_sorted = sorted(report, key=lambda r: sev_order.get(r.severity, 9))

    for idx, ar in enumerate(report_sorted, start=1):
        lines.append("")
        lines.append(f"  [{idx}] {ar.severity}  —  {ar.type}")
        lines.append(f"      Time     : {ar.timestamp}")
        lines.append(f"      Detail   : {ar.description}")
        if ar.details:
            for k, v in ar.details.items():
                if k == "sample_messages":
                    for sm in v:
                        lines.append(f"         • {sm[:140]}")
                else:
                    lines.append(f"      {k:<14}: {v}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def _format_json(report: list[AnomalyReport], entries: list[LogEntry]) -> str:
    levels = Counter(e.level for e in entries)
    timestamps = [e.timestamp for e in entries if e.timestamp]
    payload = {
        "summary": {
            "total_entries": len(entries),
            "time_range_start": min(timestamps).isoformat() if timestamps else None,
            "time_range_end": max(timestamps).isoformat() if timestamps else None,
            "anomaly_count": len(report),
            "level_distribution": dict(levels),
        },
        "anomalies": [asdict(r) for r in report],
    }
    return json.dumps(payload, indent=2, default=str)


def _format_csv(report: list[AnomalyReport]) -> str:
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "type", "severity", "description", "details"])
    for ar in report:
        writer.writerow([ar.timestamp, ar.type, ar.severity,
                         ar.description, json.dumps(ar.details)])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loganomaly",
        description=(
            "Log File Anomaly Detector — analyse one or more log files and "
            "identify statistical anomalies in event frequency, error rates, "
            "rare message patterns, and IP behaviour. Designed for cloud/DevOps "
            "workflows."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input", nargs="+", metavar="FILE",
        help="One or more log file paths to analyse.",
    )

    parser.add_argument(
        "--window", "-w", type=int, default=60,
        dest="window_seconds",
        help="Sliding-window size in seconds for frequency/error-rate analysis.",
    )

    parser.add_argument(
        "--z-threshold", "-z", type=float, default=2.0,
        help="Z-score magnitude required to flag an anomaly.",
    )

    parser.add_argument(
        "--min-windows", "-m", type=int, default=5,
        help="Minimum number of windows required before running statistical tests.",
    )

    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--strategy", "-s",
        choices=["frequency", "error_rate", "pattern_rarity", "ip_anomaly", "all"],
        default="all",
        help="Which anomaly detection strategies to run.",
    )
    strategy_group.add_argument(
        "--no-ip", action="store_true", dest="exclude_ip",
        help="Exclude IP-based anomaly detection.",
    )
    strategy_group.add_argument(
        "--no-pattern", action="store_true", dest="exclude_pattern",
        help="Exclude rare-pattern anomaly detection.",
    )

    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path.  Defaults to stdout.",
    )

    fmt_group = parser.add_mutually_exclusive_group()
    fmt_group.add_argument(
        "--format", "-f", choices=["text", "json", "csv"], default="text",
        help="Output format.",
    )
    fmt_group.add_argument(
        "--json", action="store_true", dest="fmt_json",
        help="Alias for --format json.",
    )

    parser.add_argument(
        "--sort-by", choices=["severity", "timestamp", "type"], default="severity",
        help="Sort anomalies in the report.",
    )

    parser.add_argument(
        "--limit", "-l", type=int, default=0,
        help="Max anomalies to show (0 = unlimited).",
    )

    parser.add_argument(
        "--severity-min", "-q",
        choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default="LOW",
        help="Only show anomalies at or above this severity.",
    )

    parser.add_argument(
        "--year", type=int, default=0,
        help="Fallback year for syslog-style timestamps that lack a year.",
    )

    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity.  -v info, -vv debug.",
    )

    return parser


def resolve_strategies(args: argparse.Namespace) -> list[str]:
    exclude = set()
    if args.exclude_ip:
        exclude.add("ip_anomaly")
    if args.exclude_pattern:
        exclude.add("pattern_rarity")
    if args.strategy == "all":
        strategies = ["frequency", "error_rate", "pattern_rarity", "ip_anomaly"]
    else:
        strategies = [args.strategy]
    return [s for s in strategies if s not in exclude]


def sort_report(report: list[AnomalyReport], by: str) -> list[AnomalyReport]:
    order = {"severity": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3},
             "timestamp": None, "type": None}
    if by == "severity":
        return sorted(report, key=lambda r: order["severity"].get(r.severity, 9))
    if by == "timestamp":
        return sorted(report, key=lambda r: r.timestamp)
    return sorted(report, key=lambda r: r.type)


def filter_report(report: list[AnomalyReport], min_severity: str,
                  limit: int) -> list[AnomalyReport]:
    sev_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    cutoff = sev_order.get(min_severity, 0)
    filtered = [r for r in report if sev_order.get(r.severity, 0) >= cutoff]
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def run() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Logging setup
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        level=log_levels.get(args.verbose, logging.WARNING),
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Parse inputs
    logging.info("Parsing %d input file(s)…", len(args.input))
    fallback_year = args.year or 0
    try:
        entries = LogParser(args.input,
                            fallback_year=fallback_year if fallback_year else None).parse_all()
    except Exception as exc:
        logging.error("Failed to parse input files: %s", exc)
        return 2

    if not entries:
        logging.warning("No log entries found in the provided file(s).")
        print("No entries to analyse.")
        return 0

    logging.info("Parsed %d entries.", len(entries))

    # Detect anomalies
    strategies = resolve_strategies(args)
    logging.info("Running strategies: %s", ", ".join(strategies))
    detector = AnomalyDetector(
        entries,
        window_seconds=args.window_seconds,
        z_threshold=args.z_threshold,
        min_windows=args.min_windows,
    )
    report = detector.detect(strategies=strategies)
    logging.info("Detected %d raw anomalies.", len(report))

    # Sort / filter
    report = sort_report(report, args.sort_by)
    report = filter_report(report, args.severity_min, args.limit)

    # Format output
    fmt = "json" if args.fmt_json else args.format
    if fmt == "json":
        output = _format_json(report, entries)
    elif fmt == "csv":
        output = _format_csv(report)
    else:
        output = _format_text(report, entries)

    # Write output
    if args.output:
        try:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
            logging.info("Report written to %s", out_path)
        except OSError as exc:
            logging.error("Could not write output: %s", exc)
            return 3
    else:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")

    return 0


def main() -> None:
    try:
        rc = run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        rc = 130
    except Exception as exc:
        logging.error("Unhandled exception: %s", exc, exc_info=True)
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
```