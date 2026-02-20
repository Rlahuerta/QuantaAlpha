"""
alerter.py — Email and Slack alerting for live trading events.

Usage::

    alerter = Alerter.from_config("configs/live.yaml")

    # On kill-switch trigger
    alerter.send_alert("kill_switch", {"daily_pnl": -32000, "limit_pct": 0.03})

    # On drawdown breach
    alerter.send_alert("drawdown_breach", {"mdd": -0.12, "threshold": 0.10})

    # On error
    alerter.send_alert("error", {"message": "IBKR connection failed"})

When ``dry_run=True`` (default) alerts are only logged — no email/Slack sent.
"""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)

# Human-readable templates per event type
_TEMPLATES: Dict[str, str] = {
    "kill_switch": (
        "⛔ KILL-SWITCH TRIGGERED\n"
        "Daily P&L ${daily_pnl:,.0f} breached the -{limit_pct:.0%} daily loss limit.\n"
        "Trading halted for today."
    ),
    "drawdown_breach": (
        "⚠️ MAX DRAWDOWN BREACH\n"
        "Current MDD {mdd:.1%} exceeded threshold -{threshold:.0%}.\n"
        "Consider reducing position sizes."
    ),
    "gate_failed": (
        "📋 GATE CHECK FAILED\n"
        "Paper trading has not yet met live-capital criteria:\n{reason}"
    ),
    "gate_passed": (
        "✅ GATE CHECK PASSED\n"
        "Paper trading criteria met after {days} days.\n"
        "Sharpe={sharpe:.2f}, MDD={mdd:.1%}.\n"
        "Ready for live capital — manual approval required."
    ),
    "error": "❌ LIVE TRADING ERROR\n{message}",
    "info": "ℹ️ {message}",
}


@dataclass
class AlertConfig:
    """Alerting configuration read from live.yaml."""

    dry_run: bool = True
    # Email
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_addr: str = ""
    to_addrs: list = field(default_factory=list)
    # Slack
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    # Prefix for all alert subjects
    subject_prefix: str = "[QuantaAlpha]"


class Alerter:
    """
    Unified alerter for email and Slack notifications.

    Parameters
    ----------
    config : AlertConfig
        Alerting settings.
    """

    def __init__(self, config: AlertConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Alerter":
        """Load from a live.yaml file."""
        data = yaml.safe_load(Path(config_path).read_text())
        raw = data.get("alerting", {})
        cfg = AlertConfig(
            dry_run=raw.get("dry_run", True),
            email_enabled=raw.get("email", {}).get("enabled", False),
            smtp_host=raw.get("email", {}).get("smtp_host", "smtp.gmail.com"),
            smtp_port=int(raw.get("email", {}).get("smtp_port", 587)),
            smtp_user=raw.get("email", {}).get("smtp_user", ""),
            smtp_password=raw.get("email", {}).get("smtp_password", ""),
            from_addr=raw.get("email", {}).get("from_addr", ""),
            to_addrs=raw.get("email", {}).get("to_addrs", []),
            slack_enabled=raw.get("slack", {}).get("enabled", False),
            slack_webhook_url=raw.get("slack", {}).get("webhook_url", ""),
            subject_prefix=raw.get("subject_prefix", "[QuantaAlpha]"),
        )
        return cls(cfg)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Alerter":
        """Create directly from a dict (for testing)."""
        cfg = AlertConfig(**{k: v for k, v in raw.items() if hasattr(AlertConfig, k)})
        return cls(cfg)

    # ------------------------------------------------------------------
    # Internal senders
    # ------------------------------------------------------------------

    def _format_message(self, event_type: str, details: Dict[str, Any]) -> str:
        template = _TEMPLATES.get(event_type, "ℹ️ {event_type}: {details}")
        try:
            return template.format(event_type=event_type, details=details, **details)
        except (KeyError, ValueError):
            return f"{event_type}: {json.dumps(details, default=str)}"

    def _send_email(self, subject: str, body: str) -> bool:  # pragma: no cover
        """Send SMTP email; returns True on success."""
        if not self.config.email_enabled or not self.config.to_addrs:
            return False
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.config.from_addr
            msg["To"] = ", ".join(self.config.to_addrs)
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as smtp:
                smtp.starttls()
                if self.config.smtp_user:
                    smtp.login(self.config.smtp_user, self.config.smtp_password)
                smtp.sendmail(self.config.from_addr, self.config.to_addrs, msg.as_string())
            log.info("Alert email sent: %s", subject)
            return True
        except Exception as exc:
            log.error("Failed to send alert email: %s", exc)
            return False

    def _send_slack(self, message: str) -> bool:  # pragma: no cover
        """Post to Slack webhook; returns True on success."""
        if not self.config.slack_enabled or not self.config.slack_webhook_url:
            return False
        try:
            payload = json.dumps({"text": message}).encode()
            req = urllib.request.Request(
                self.config.slack_webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            log.info("Slack alert sent")
            return True
        except urllib.error.URLError as exc:
            log.error("Failed to send Slack alert: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Format and dispatch an alert.

        Parameters
        ----------
        event_type : str
            One of: kill_switch, drawdown_breach, gate_failed, gate_passed,
            error, info — or any custom string.
        details : dict, optional
            Key-value pairs interpolated into the message template.

        Returns
        -------
        bool
            True if at least one channel delivered the alert (or dry_run).
        """
        details = details or {}
        message = self._format_message(event_type, details)
        subject = f"{self.config.subject_prefix} {event_type.replace('_', ' ').title()}"

        log.warning("ALERT [%s]: %s", event_type, message)

        if self.config.dry_run:
            log.info("Alerter dry_run=True — not sending email/Slack")
            return True  # count as success in dry-run

        sent = False
        if self.config.email_enabled:
            sent |= self._send_email(subject, message)
        if self.config.slack_enabled:
            sent |= self._send_slack(message)
        return sent
