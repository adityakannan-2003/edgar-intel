"""edgar-intel: retrieval, evaluation and agent stack over SEC EDGAR filings.

The organising idea is that every quality claim in this repo traces back to
something externally verifiable. Numeric answers are checked against XBRL facts
the SEC itself publishes; narrative answers are scored by an LLM judge whose
agreement with human labels is measured and reported as Cohen's kappa. Nothing
reports a number it cannot justify.
"""

__version__ = "0.1.0"
