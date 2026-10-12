"""Best-effort Forex Factory economic-calendar reader.

Forex Factory does not provide a documented public calendar API. This module
reads the public calendar HTML, requests UTC via the site's timezone cookie,
and fails explicitly when the page layout or access changes. Do not treat a
successful HTTP response as proof that the site's timezone cookie was honored.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup

CALENDAR_URL = "https://www.forexfactory.com/calendar"
TIMEOUT_SECONDS = 15
USER_AGENT = (
    "Mozilla/5.0 (compatible; SSADCalendar/1.0; "
    "+https://github.com/Bluxx283/smart-session-trading-bot)"
)


class CalendarFetchError(RuntimeError):
    """Raised when the public calendar cannot be fetched or parsed."""


def _cell_text(row: Any, selector: str) -> str:
    cell = row.select_one(selector)
    return cell.get_text(" ", strip=True) if cell else ""


def _impact(row: Any) -> str:
    for node in row.select("[class]"):
        classes = " ".join(node.get("class", []))
        if "ff-impact-red" in classes:
            return "High"
        if "ff-impact-ora" in classes:
            return "Medium"
        if "ff-impact-yel" in classes:
            return "Low"
        if "ff-impact-gra" in classes:
            return "Holiday"
    return "Unknown"


def _parse_date(raw: str, today: date) -> date | None:
    raw = " ".join((raw or "").split())
    if not raw:
        return None
    for fmt in ("%a %b %d", "%b %d", "%A %B %d"):
        try:
            parsed = datetime.strptime(raw, fmt).date().replace(year=today.year)
            if (parsed - today).days > 180:
                parsed = parsed.replace(year=today.year - 1)
            elif (today - parsed).days > 180:
                parsed = parsed.replace(year=today.year + 1)
            return parsed
        except ValueError:
            continue
    return None


def _parse_time(raw: str) -> time | None:
    raw = " ".join((raw or "").split()).lower().replace(" ", "")
    if not raw or raw in {"all-day", "allday", "tentative", "day1", "day2", "day3", "holiday"}:
        return None
    for fmt in ("%I:%M%p", "%I%p", "%H:%M"):
        try:
            return datetime.strptime(raw, fmt).time()
        except ValueError:
            continue
    return None


def parse_calendar_html(html: str, *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Parse calendar rows into normalized events; timestamps are interpreted as UTC."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.calendar__row")
    if not rows:
        raise CalendarFetchError(
            "Forex Factory returned no recognizable calendar rows; its page may have changed "
            "or access may be blocked."
        )

    events: list[dict[str, Any]] = []
    active_date: date | None = None
    for row in rows:
        parsed_date = _parse_date(_cell_text(row, "td.calendar__date"), today)
        if parsed_date:
            active_date = parsed_date
        if active_date is None:
            continue

        currency = _cell_text(row, "td.calendar__currency").upper()
        title = _cell_text(row, "td.calendar__event span.calendar__event-title")
        if not title:
            title = _cell_text(row, "td.calendar__event")
        if not title or not currency:
            continue

        raw_time = _cell_text(row, "td.calendar__time")
        parsed_time = _parse_time(raw_time)
        event_dt = datetime.combine(active_date, parsed_time, tzinfo=timezone.utc) if parsed_time else None
        events.append({
            "id": "|".join((event_dt.isoformat() if event_dt else active_date.isoformat(), currency, title)),
            "datetime_utc": event_dt.isoformat() if event_dt else None,
            "date_utc": active_date.isoformat(),
            "time_source": raw_time or "Time not announced",
            "currency": currency,
            "impact": _impact(row),
            "event": title,
            "actual": _cell_text(row, "td.calendar__actual"),
            "forecast": _cell_text(row, "td.calendar__forecast"),
            "previous": _cell_text(row, "td.calendar__previous"),
            "url": CALENDAR_URL,
        })

    if not events:
        raise CalendarFetchError(
            "The calendar page loaded, but no usable events were parsed. "
            "The source layout may have changed."
        )
    return events


def fetch_calendar(*, now: datetime | None = None) -> list[dict[str, Any]]:
    """Fetch the current calendar view; no data is fabricated on errors."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    # Request UTC explicitly; if the source ignores this cookie, times may be misaligned.
    session.cookies.set("fftimezone", "UTC", domain=".forexfactory.com")
    try:
        response = session.get(CALENDAR_URL, params={"week": "this"}, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CalendarFetchError(f"Could not fetch Forex Factory calendar: {exc}") from exc

    if "captcha" in response.url.lower() or "verify you are human" in response.text.lower():
        raise CalendarFetchError(
            "Forex Factory is presenting an access-verification page. "
            "Calendar data is unavailable; no events were substituted."
        )
    return parse_calendar_html(response.text, now=now)
