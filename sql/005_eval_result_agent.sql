-- Per-case agent record for runs of the tool-calling agent on the golden set
-- (`edgar-intel eval agent`): outcome, steps, tools called, the draft an
-- escalation held back, and the split between agent and judge cost. NULL on
-- every retrieve-then-answer run.
ALTER TABLE eval_results
    ADD COLUMN IF NOT EXISTS agent JSONB;
