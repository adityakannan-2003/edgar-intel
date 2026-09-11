"""SEC EDGAR HTTP client.

Two things here are non-negotiable and both are SEC policy rather than style
preference: every request carries a descriptive User-Agent with a contact
address, and requests are rate-limited to under 10/second. Violating either
gets your IP blocked, so the limiter is enforced in the client itself rather
than left to callers to remember.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
from ..config import get_settings
from ..retry import retry


class RateLimiter:
    """Simple token-bucket. Thread-safe because ingestion fans out."""

    def __init__(self, rate_per_sec: float) -> None:
        self.min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last = now


class EdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        s = get_settings()
        ua = user_agent or s.user_agent
        if "@" not in ua:
            raise ValueError(
                "EDGAR_USER_AGENT must include a contact email address -- the SEC "
                "requires it and will block requests without one."
            )
        self.settings = s
        self.limiter = RateLimiter(s.edgar_rate_limit_rps)
        self._client = httpx.Client(
            headers={
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        exceptions=(httpx.TransportError, httpx.HTTPStatusError),
        attempts=4,
        initial=2.0,
        maximum=30.0,
    )
    def _get(self, url: str) -> httpx.Response:
        self.limiter.acquire()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------ API
    def ticker_map(self) -> dict[str, dict[str, Any]]:
        """Ticker -> {cik, title}. CIKs are zero-padded to 10 digits, which is
        the form every other EDGAR endpoint expects."""
        url = f"{self.settings.edgar_www}/files/company_tickers.json"
        data = self._get(url).json()
        out: dict[str, dict[str, Any]] = {}
        for row in data.values():
            ticker = str(row["ticker"]).upper()
            out[ticker] = {
                "cik": str(row["cik_str"]).zfill(10),
                "title": row["title"],
            }
        return out

    def submissions(self, cik: str) -> dict[str, Any]:
        url = f"{self.settings.edgar_base}/submissions/CIK{cik}.json"
        return self._get(url).json()

    def company_facts(self, cik: str) -> dict[str, Any]:
        """All XBRL facts the company has ever reported. This is the ground
        truth the numeric evaluation set is generated from."""
        url = f"{self.settings.edgar_base}/api/xbrl/companyfacts/CIK{cik}.json"
        return self._get(url).json()

    def filing_document(self, cik: str, accession: str, primary_doc: str) -> str:
        acc_nodash = accession.replace("-", "")
        cik_int = str(int(cik))
        url = f"{self.settings.edgar_www}/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"
        return self._get(url).text

    def filing_url(self, cik: str, accession: str, primary_doc: str) -> str:
        acc_nodash = accession.replace("-", "")
        cik_int = str(int(cik))
        return f"{self.settings.edgar_www}/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"


def recent_filings(
    submissions: dict[str, Any], forms: list[str], limit: int = 5
) -> list[dict[str, Any]]:
    """Flatten the columnar `filings.recent` block into row dicts.

    EDGAR returns parallel arrays rather than records, which is compact on the
    wire and awkward everywhere else.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []
    keys = [
        "accessionNumber",
        "form",
        "filingDate",
        "reportDate",
        "primaryDocument",
    ]
    cols = {k: recent.get(k, []) for k in keys}
    n = len(cols["accessionNumber"])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        form = cols["form"][i]
        if form not in forms:
            continue
        rows.append(
            {
                "accession": cols["accessionNumber"][i],
                "form": form,
                "filing_date": cols["filingDate"][i] or None,
                "period_end": cols["reportDate"][i] or None,
                "primary_doc": cols["primaryDocument"][i],
            }
        )
        if len(rows) >= limit:
            break
    return rows
