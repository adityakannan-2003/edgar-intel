# Architecture and the decisions behind it

Every section below is a decision someone will ask you to justify. The reasoning
matters more than the choice — a defensible "we chose X because Y, and the
tradeoff is Z" beats a correct answer you cannot explain.

---

## 1. Why SEC filings

Three properties, in order of importance.

**Verifiable ground truth.** The SEC publishes XBRL company facts — the same
figures that appear in the filing text, as structured data. That gives the
evaluation an external anchor. Almost no other freely available corpus has this.
Without it, you are grading an LLM with an LLM and calling the result quality.

**Genuinely hard retrieval.** A 10-K is 100+ pages of nested tables, footnotes
and page furniture. Chunking choices have visible, measurable consequences,
which makes the strategy comparison a real experiment rather than a formality.

**Legible to the reader.** A hiring manager at a fintech, an enterprise software
company, or a bank immediately understands the domain. A retrieval system over
your Notion notes does not travel.

## 2. Why Postgres + pgvector, not a dedicated vector database

The corpus here is tens of thousands of chunks. Purpose-built engines (Qdrant,
Milvus) start earning their operational cost above roughly 50M vectors, where
index build time and tail latency under concurrent writes degrade for pgvector.
Below that, Postgres wins on three counts:

- **One store.** Vectors, full-text index, XBRL facts and eval results in one
  transactional database. A single query filters by company and fiscal year
  *and* ranks by vector distance. With a separate vector store that becomes two
  round trips and a consistency problem.
- **No second system to operate.** One backup, one connection pool, one set of
  credentials.
- **ACID.** Re-indexing a strategy is a transaction, not a partial-write hazard.

Naming a specific threshold and the reason for it is the difference between
judgment and cargo-culting. *If* the corpus grew past ~50M vectors, the
migration path is `retrieval/search.py` — the interface is already isolated.

## 3. Why hybrid retrieval

Dense retrieval generalises: it finds "the company's borrowing costs rose" when
you asked about interest expense. It is systematically weak on exactly the
tokens that matter in a filing — tickers, tag names, specific dollar amounts,
years — because a 512-token passage embedding does not preserve a single rare
literal well.

Lexical search is the mirror image: exact on literals, useless on paraphrase.

A real question — *"what did NVDA report for R&D in fiscal 2024"* — contains a
rare literal **and** a paraphrase. You need both retrievers.

### Why RRF and not weighted score fusion

Cosine distances and `ts_rank_cd` scores live on incomparable scales. Any
attempt to normalise them into a weighted sum needs constants that have to be
re-tuned whenever either retriever changes. Reciprocal Rank Fusion uses only
rank position:

```
score(d) = Σ  weight_i / (rrf_k + rank_i(d))
```

No calibration, degrades gracefully, and `rrf_k = 60` is the constant from the
original paper — left untuned deliberately, because tuning it against the same
evaluation set used for reporting would leak the test set into the system.

## 4. Why a cross-encoder reranker, and why only over the top-k

The bi-encoder that built the index embeds query and passage **independently**,
so it can never model their interaction. A cross-encoder reads both together and
is far more accurate — and far too slow to run over a corpus.

So it runs over the ~50 candidates the cheap retriever already found. That
two-stage shape is the entire reason reranking is worth its latency, and the
stages are timed separately (`stage_latency_ms`) so the budget stays legible.

**Worth knowing:** reranking does not improve recall@k when k covers the whole
candidate set — it only moves hits upward. That is why nDCG is reported
alongside recall. A project reporting only recall would conclude reranking did
nothing. There is a test asserting exactly this
(`test_recall_is_blind_to_order_but_ndcg_is_not`).

## 5. Why four chunking strategies

Not to use four. To be able to say which one is better *on this corpus* and show
the number.

| Strategy | What it does | Expected failure on filings |
|---|---|---|
| `fixed` | Character budget, no structure | Cuts tables in half — a number in one chunk, its row label in the next |
| `recursive` | Paragraph → sentence → character | Respects prose, still crosses Item boundaries |
| `section_aware` | Never crosses an Item, prefixes the section title | The prefix puts "Risk Factors" into the embedded text |
| `semantic` | Groups sentences by embedding similarity | Best coherence, one embedding call per sentence, usually a small gain over section-aware here |

All four are stored side by side under their own `strategy` label, so a run
points at one and results compare apples to apples: identical source text,
identical embedding model, identical questions, one variable.

## 6. Why the evaluation is split numeric / narrative

| | Numeric | Narrative |
|---|---|---|
| Source of truth | XBRL fact from the SEC | Human-written reference |
| Grading | Numeric comparison, relative tolerance | LLM judge |
| Trust | High — externally verified | Only as good as the judge's κ |
| Generation | Automatic, hundreds of cases | Hand-written |

Auto-generating the numeric cases is what makes a 200-case set achievable in an
afternoon. Hand-writing 200 questions is what stops most people from ever
building an eval set at all.

The overall score is weighted by case count so adding narrative cases cannot
quietly swamp the verifiable numeric signal.

### The evidence-linking trap

`link_evidence` finds relevant chunks by searching for the **expected answer's**
distinctive tokens — not the question. Searching the question would label
whatever the retriever already returns as "relevant", making recall@k trivially
1.0 and the metric meaningless.

This is the single easiest way to build an eval set that flatters your system,
and it is worth being able to say you avoided it on purpose.

## 7. Why the judge is calibrated

An LLM judge that is never checked against a human is a number generator. Raw
agreement cannot distinguish a good judge from one that says "correct" to
everything on a set where 90% of answers happen to be correct. Cohen's κ
corrects for chance agreement and can.

Below κ = 0.60 the report states the narrative scores are untrustworthy and the
gate fails. Twenty labels is the noise floor; forty makes the estimate stable.

Labels are matched on `(case_id, answer_hash)`, not `case_id` — a human label
applies to the *specific answer* that was labelled. Reusing it after the answer
changed would silently inflate κ.

## 8. Why the agent is bounded the way it is

| Bound | Failure it contains |
|---|---|
| Step ceiling | Infinite tool loops |
| Retry budget | Malformed output burning the whole budget |
| Loop detection | Identical repeated calls — the most expensive common failure |
| Confidence floor | Answering when it should hand off |
| Citation check | Cited passage ids no tool returned |
| Numeric grounding | Figures that appeared in no tool output |
| Cost ceiling | A runaway run costing real money |

On exhaustion the agent returns what it has with `outcome="exhausted"` — never
an invented answer. Every step is recorded, so a trace answers "what did it do
and why" without re-running anything.

### Why six tools

Every additional tool enlarges the selection space and measurably degrades
choice accuracy. A tool that is rarely the right call is a tool that is often
the wrong one. `escalate_to_human` is a first-class tool because giving the
model an explicit, cheap way to stop is what keeps it from inventing an answer
when the evidence is not there.

`list_coverage` exists so the agent can discover the boundary of its own
knowledge. Most agent-layer hallucinations are really coverage questions the
agent had no way to ask.

## 9. Why MCP

Not because it is fashionable. Because the tool definitions in `tools.py` become
the single source of truth for three consumers — the agent loop, the HTTP API,
and any MCP client — instead of being reimplemented once per surface. Schemas
are generated from the same Pydantic models the agent validates against, so they
cannot drift. Adding a tool means editing one file.

## 10. Why fine-tune for extraction, not analysis

Fine-tuning a small model to write good narrative analysis is a bad bet: output
is open-ended, grading needs a judge, and a frontier model will beat you.

Fine-tuning for **structured extraction** is a good bet: output is short and
schema-constrained, correctness is checkable against XBRL without a judge, and
small models close most of the gap on narrow extraction at a fraction of the
cost.

Two details that decide whether the result is real:

- **Negatives.** ~25% of training examples are passages with no financial facts,
  labelled `[]`. Without them the model learns "always emit facts" and
  hallucinates numbers on prose. This is the most commonly omitted step in
  extraction fine-tunes and it shows up immediately as precision collapse.
- **Split by company, not by row.** Random splitting puts chunks from the same
  filing on both sides, and the held-out score is inflated by memorised
  boilerplate.

### Why LoRA

Full fine-tuning of an 8B model needs roughly 60–80GB of optimiser state and
gradients. LoRA trains low-rank matrices per attention projection — well under
1% of parameters — so it fits on a single 24GB card, trains in hours, and the
adapter is hundreds of megabytes instead of sixteen gigabytes. You can hold
several adapters and swap them at load time.

`r=16 / alpha=32` keeps `alpha/r = 2`, so the effective update scale stays
stable if you change `r` later.

## 11. Why liveness and readiness are separate endpoints

`/health` answers "is the process up". `/ready` checks the database and whether
the index actually has embeddings. An orchestrator that restarts the process
because the database blinked has made an outage worse, not better.

## 12. Why the benchmark sweeps concurrency

A latency number from a single-threaded loop is the one number guaranteed not to
describe production. The sweep finds the **knee** — where p95 starts rising
faster than throughput grows — and that number is your per-instance capacity and
the honest answer to "how many users can this handle".

Throughput counts successes only. Counting failed requests as throughput is how
a broken service benchmarks well.

---

## Known limitations

Being able to state these is worth more than pretending they don't exist.

1. **`ts_rank_cd` is not BM25.** No document-length saturation. ParadeDB's
   `pg_search` would provide true BM25; it is a one-function change.
2. **Fine-tuning labels are distant supervision.** A chunk containing "383,285"
   is *probably* stating total net sales, but might state something else that
   happens to equal it. The eval set is generated independently from XBRL rather
   than from these labels, which limits the damage.
3. **Narrative reference answers ship generic.** They are placeholders and say
   so. Replacing them with what the filings actually say is step one of making
   this yours.
4. **Section parsing is regex-based.** It handles the major filers; unusual
   filings will need adjustment. The `find_item_offsets` "last occurrence" rule
   handles the table-of-contents collision, which is the subtle one.
5. **Semantic chunking cost is not amortised.** One embedding call per sentence
   at build time. Fine for this corpus, expensive at scale.
6. **No incremental re-indexing.** Rebuilding a strategy re-chunks every filing.
   Embedding is resumable (`embed_pending`), chunking is not.
