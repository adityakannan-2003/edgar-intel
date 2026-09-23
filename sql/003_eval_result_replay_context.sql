-- Persist exact inputs required to replay narrative judge decisions.
ALTER TABLE eval_results
    ADD COLUMN IF NOT EXISTS question TEXT;

ALTER TABLE eval_results
    ADD COLUMN IF NOT EXISTS context_text TEXT;
