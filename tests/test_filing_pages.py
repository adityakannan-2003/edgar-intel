"""Following the older pages of EDGAR's submissions JSON.

`filings.recent` covers about a year. For a bank that files tens of thousands
of prospectuses a year, that year holds one 10-K, and `ingest run --years 3`
used to stop there. These tests feed a submissions JSON shaped that way, with
the 10-Ks in an older page, and never touch the network.
"""

from __future__ import annotations

import httpx
import pytest

from edgar_intel.ingest.edgar_client import EdgarClient, recent_filings


def _block(rows: list[tuple[str, str, str, str]]) -> dict[str, list[str]]:
    """Columnar block, the way EDGAR serves both `recent` and each page."""
    return {
        "accessionNumber": [r[0] for r in rows],
        "form": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows],
        "reportDate": [r[3] for r in rows],
        "primaryDocument": [f"{r[0]}.htm" for r in rows],
    }


@pytest.fixture
def frequent_filer() -> tuple[dict, dict[str, dict]]:
    """A recent block of prospectuses and no 10-K, then two older pages."""
    submissions = {
        "cik": "0000019617",
        "filings": {
            "recent": _block(
                [
                    ("0000000001-26-000003", "424B2", "2026-10-15", ""),
                    ("0000000001-26-000002", "FWP", "2026-06-01", ""),
                    ("0000000001-25-000009", "424B2", "2025-10-16", ""),
                ]
            ),
            "files": [
                {
                    "name": "CIK0000019617-submissions-001.json",
                    "filingCount": 5,
                    "filingFrom": "2023-02-01",
                    "filingTo": "2025-10-14",
                },
                {
                    "name": "CIK0000019617-submissions-002.json",
                    "filingCount": 2,
                    "filingFrom": "2021-02-01",
                    "filingTo": "2023-01-31",
                },
            ],
        },
    }
    pages = {
        "CIK0000019617-submissions-001.json": _block(
            [
                ("0000000001-25-000008", "424B2", "2025-10-14", ""),
                ("0000000001-25-000001", "10-K", "2025-02-14", "2024-12-31"),
                ("0000000001-24-000005", "10-Q", "2024-11-01", "2024-09-30"),
                ("0000000001-24-000001", "10-K", "2024-02-16", "2023-12-31"),
                ("0000000001-23-000001", "10-K", "2023-02-21", "2022-12-31"),
            ]
        ),
        "CIK0000019617-submissions-002.json": _block(
            [
                ("0000000001-22-000001", "10-K", "2022-02-22", "2021-12-31"),
                ("0000000001-21-000001", "10-K", "2021-02-23", "2020-12-31"),
            ]
        ),
    }
    return submissions, pages


def _fetcher(pages: dict[str, dict]):
    fetched: list[str] = []

    def fetch(name: str) -> dict:
        fetched.append(name)
        return pages[name]

    return fetch, fetched


def test_the_recent_block_alone_finds_no_10k(frequent_filer):
    submissions, _ = frequent_filer
    assert recent_filings(submissions, ["10-K"], limit=3) == []


def test_three_10ks_come_from_the_older_page_newest_first(frequent_filer):
    submissions, pages = frequent_filer
    fetch, fetched = _fetcher(pages)

    rows = recent_filings(submissions, ["10-K"], limit=3, fetch_page=fetch)

    assert [r["accession"] for r in rows] == [
        "0000000001-25-000001",
        "0000000001-24-000001",
        "0000000001-23-000001",
    ]
    assert rows[0] == {
        "accession": "0000000001-25-000001",
        "form": "10-K",
        "filing_date": "2025-02-14",
        "period_end": "2024-12-31",
        "primary_doc": "0000000001-25-000001.htm",
    }
    # Three were found on the first page, so the second is never requested.
    assert fetched == ["CIK0000019617-submissions-001.json"]


def test_paging_continues_until_the_limit_is_met(frequent_filer):
    submissions, pages = frequent_filer
    fetch, fetched = _fetcher(pages)

    rows = recent_filings(submissions, ["10-K"], limit=4, fetch_page=fetch)

    assert [r["filing_date"] for r in rows] == [
        "2025-02-14",
        "2024-02-16",
        "2023-02-21",
        "2022-02-22",
    ]
    assert fetched == [
        "CIK0000019617-submissions-001.json",
        "CIK0000019617-submissions-002.json",
    ]


def test_no_page_is_fetched_when_recent_already_has_enough(frequent_filer):
    submissions, pages = frequent_filer
    fetch, fetched = _fetcher(pages)

    rows = recent_filings(submissions, ["424B2"], limit=2, fetch_page=fetch)

    assert [r["accession"] for r in rows] == ["0000000001-26-000003", "0000000001-25-000009"]
    assert fetched == []


def test_running_out_of_pages_returns_what_was_found(frequent_filer):
    submissions, pages = frequent_filer
    fetch, fetched = _fetcher(pages)

    rows = recent_filings(submissions, ["10-K"], limit=10, fetch_page=fetch)

    assert len(rows) == 5
    assert len(fetched) == 2


def test_a_page_goes_through_the_rate_limiter_with_the_user_agent():
    client = EdgarClient()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_block([]))

    client._client = httpx.Client(
        headers=client._client.headers, transport=httpx.MockTransport(handler)
    )
    acquired: list[None] = []
    real_acquire = client.limiter.acquire
    client.limiter.acquire = lambda: (acquired.append(None), real_acquire())[1]

    with client:
        page = client.submissions_page("CIK0000019617-submissions-001.json")

    assert page["accessionNumber"] == []
    assert str(seen[0].url) == (
        f"{client.settings.edgar_base}/submissions/CIK0000019617-submissions-001.json"
    )
    assert seen[0].headers["User-Agent"] == client.settings.user_agent
    assert len(acquired) == 1
