"""Turn a 10-K HTML document into clean, section-labelled text.

This module is where most of the real difficulty in this project lives, and it
is worth saying why out loud, because it is the part that separates a working
retrieval system from a tutorial one.

A 10-K is not prose. It is 100+ pages of nested tables, footnote markers,
page furniture, and inline XBRL tags. Naive text extraction produces a soup in
which a revenue figure and the label three cells away from it end up hundreds
of characters apart -- so a chunk that contains the number no longer contains
what the number means, and retrieval quietly fails in a way that looks like a
model problem.

What is done about it:

  * tables are flattened row-wise with cell separators preserved, so the label
    and the value stay adjacent in the text stream;
  * page furniture and the table of contents are dropped before chunking;
  * Item boundaries are detected so `section_aware` chunking never straddles
    the line between Risk Factors and MD&A.

The naive strategies deliberately do none of this, which is what makes the
chunking comparison in the evaluation a fair fight rather than a strawman.

Built on the standard library's html.parser rather than lxml or selectolax.
Filings are a few megabytes and parsed once at ingest time, so the speed
difference is irrelevant, and a zero-dependency parser is one less thing
between someone and running this repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser as _StdHTMLParser

# Items we care about in a 10-K. Order matters: it defines section boundaries.
ITEM_PATTERNS: list[tuple[str, str]] = [
    ("1", r"item\s*1\s*[\.\-–—:]?\s*business"),
    ("1A", r"item\s*1a\s*[\.\-–—:]?\s*risk\s*factors"),
    ("1B", r"item\s*1b\s*[\.\-–—:]?\s*unresolved\s*staff\s*comments"),
    ("2", r"item\s*2\s*[\.\-–—:]?\s*propert"),
    ("3", r"item\s*3\s*[\.\-–—:]?\s*legal\s*proceedings"),
    ("5", r"item\s*5\s*[\.\-–—:]?\s*market\s*for"),
    ("7", r"item\s*7\s*[\.\-–—:]?\s*management.s\s*discussion"),
    ("7A", r"item\s*7a\s*[\.\-–—:]?\s*quantitative\s*and\s*qualitative"),
    ("8", r"item\s*8\s*[\.\-–—:]?\s*financial\s*statements"),
    ("9A", r"item\s*9a\s*[\.\-–—:]?\s*controls\s*and\s*procedures"),
]

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")
_PAGE_FURNITURE = re.compile(
    r"^\s*(table of contents|page\s*\|?\s*\d+|\d+\s*\|\s*\d{4}\s*form\s*10-k)\s*$",
    re.I | re.M,
)

# Tags whose content is markup or styling, never readable text.
_DROP_TAGS = {"script", "style", "head", "noscript", "title", "meta", "link"}
# Tags that should force a line break in the flattened output.
_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer", "table", "thead", "tbody",
}


class _FilingTextExtractor(_StdHTMLParser):
    """Flatten filing HTML, keeping table rows on one line.

    The cell separator is the whole point. Joining cells with ' | ' keeps
    "Total net sales | 383,285 | 394,328" together, so a chunk containing the
    figure also contains its label. Emitting cells on separate lines -- which
    is what a naive text extraction does -- is the single most damaging thing
    you can do to retrieval over financial filings.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppress_depth = 0
        self._in_cell = False
        self._cells: list[str] = []
        self._cell_buf: list[str] = []
        self._in_row = False

    # -- helpers
    def _emit(self, text: str) -> None:
        if text:
            self.parts.append(text)

    def _newline(self) -> None:
        if self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _DROP_TAGS:
            self._suppress_depth += 1
            return
        if self._suppress_depth:
            return
        if tag == "tr":
            self._in_row = True
            self._cells = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_buf = []
        elif tag in _BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_TAGS:
            self._suppress_depth = max(0, self._suppress_depth - 1)
            return
        if self._suppress_depth:
            return
        if tag in ("td", "th"):
            cell = _WS.sub(" ", "".join(self._cell_buf)).strip()
            if cell:
                self._cells.append(cell)
            self._in_cell = False
            self._cell_buf = []
        elif tag == "tr":
            if self._cells:
                self._newline()
                self._emit(" | ".join(self._cells))
                self._newline()
            self._cells = []
            self._in_row = False
        elif tag in _BLOCK_TAGS:
            self._newline()

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag == "br" and not self._suppress_depth:
            if self._in_cell:
                self._cell_buf.append(" ")
            else:
                self._newline()

    def handle_data(self, data: str) -> None:
        if self._suppress_depth or not data.strip():
            # Preserve a single space so adjacent inline elements do not fuse
            # into one word ("netsales" instead of "net sales").
            if data and not self._suppress_depth:
                if self._in_cell:
                    self._cell_buf.append(" ")
                elif self.parts and not self.parts[-1].endswith((" ", "\n")):
                    self.parts.append(" ")
            return
        if self._in_cell:
            self._cell_buf.append(data)
        else:
            self._emit(data)

    def text(self) -> str:
        return "".join(self.parts)


@dataclass(slots=True)
class Section:
    item: str | None
    title: str
    ordinal: int
    body: str

    @property
    def char_len(self) -> int:
        return len(self.body)


def html_to_text(html: str) -> str:
    """Flatten filing HTML to text with table structure preserved."""
    parser = _FilingTextExtractor()
    # Filings occasionally contain malformed markup; html.parser is lenient,
    # but an unterminated construct at EOF still needs closing out.
    parser.feed(html)
    parser.close()

    text = parser.text().replace(" ", " ")
    text = _WS.sub(" ", text)
    text = _PAGE_FURNITURE.sub("", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def find_item_offsets(text: str) -> list[tuple[str, int]]:
    """Locate Item headings while preserving canonical 10-K order.

    Filings often mention Item headings multiple times: in the table of
    contents, at the real section heading, and later in cross-references.

    Walk backward through the expected Item order and choose the latest match
    that still appears before the next selected Item. This prefers body
    headings over the table of contents without allowing a later prose
    cross-reference to reorder sections.
    """
    selected: list[tuple[str, int]] = []
    upper_bound = len(text) + 1

    for item, pattern in reversed(ITEM_PATTERNS):
        matches = list(re.finditer(pattern, text, flags=re.I))
        if not matches:
            continue

        valid = [match for match in matches if match.start() < upper_bound]
        if not valid:
            continue

        chosen = valid[-1]
        selected.append((item, chosen.start()))
        upper_bound = chosen.start()

    selected.reverse()
    return selected


def split_sections(text: str, min_chars: int = 400) -> list[Section]:
    """Split flattened filing text into Item-labelled sections.

    Every located Item becomes a section regardless of length. A short section
    is not a parse failure: filers routinely satisfy an Item by *incorporation
    by reference*, which is standard practice and perfectly legal. Real
    examples from the corpus:

      NVIDIA Item 8  (305 chars): "The information required by this Item is set
                                   forth in our Consolidated Financial
                                   Statements and Notes thereto..."
      P&G   Item 7A  (274 chars): "...incorporated by reference to the section
                                   entitled Other Information in the MD&A..."

    An earlier version dropped sections below `min_chars`. That silently threw
    the text away *and* lost the fact that the Item existed -- and since
    retrieval filters on `item`, a missing Item is a query that can never
    match. Cheap to keep, expensive to lose.

    `min_chars` now applies only to the cover block before the first Item,
    which genuinely is page furniture when it is short.
    """
    offsets = find_item_offsets(text)
    if not offsets:
        return [Section(item=None, title="Full Document", ordinal=0, body=text)]

    sections: list[Section] = []
    preamble = text[: offsets[0][1]].strip()
    ordinal = 0
    if len(preamble) >= min_chars:
        sections.append(Section(item=None, title="Cover", ordinal=ordinal, body=preamble))
        ordinal += 1

    for idx, (item, start) in enumerate(offsets):
        end = offsets[idx + 1][1] if idx + 1 < len(offsets) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        sections.append(
            Section(
                item=item,
                title=ITEM_TITLES.get(item, f"Item {item}"),
                ordinal=ordinal,
                body=body,
            )
        )
        ordinal += 1
    return sections


def parse_filing(html: str) -> list[Section]:
    return split_sections(html_to_text(html))
