"""The operational surface of a deployed instance.

Two defects motivate this file, both of the same shape: a check that reports
healthy while the thing it is supposed to check is broken.

  `/ready` verified the database and the index and not the embedder. The shipped
  Dockerfile installed `.[serving,obs]` and not `.[models]`, so `LocalEmbedder`
  raised `ImportError` on the first query. A container in which `/ready` returned
  200 and every `/search` in dense or hybrid mode returned 500 -- and an
  orchestrator watching `/ready` would have called that a successful deploy.

  `/ask` runs the agent against a paid model using the deployment's own key, and
  had no authentication and no rate limit. Fine on localhost, a standing invoice
  on a public URL.

The controls are inert by default, which is deliberate -- the test suite,
docker-compose and local development must not need a key -- and inert-by-default
is exactly why they need tests that assert both states.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="serving extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

from edgar_intel.serving import app as app_module  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.db, "healthcheck", lambda: True)
    monkeypatch.setattr(
        app_module.db, "query_one", lambda *a, **k: {"n": 1234}, raising=False
    )
    app_module.reset_rate_limits()
    with TestClient(app_module.app) as c:
        yield c


def _embedder(vec):
    class E:
        def embed(self, texts):
            return vec

    return E()


class TestReadinessChecksTheQueryPath:
    def test_ready_when_database_index_and_embedder_all_work(self, client, monkeypatch):
        monkeypatch.setattr(
            "edgar_intel.providers.get_embedder", lambda: _embedder([[0.0] * 384])
        )
        r = client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["checks"]["embedder"] is True
        assert body["checks"]["embedder_dim"] == 384
        assert body["checks"]["indexed_chunks"] == 1234

    def test_a_missing_models_extra_makes_the_instance_not_ready(self, client, monkeypatch):
        """The exact production defect: the image that cannot embed a query."""

        def boom():
            raise ImportError("Local embeddings need the 'models' extra")

        monkeypatch.setattr("edgar_intel.providers.get_embedder", boom)
        r = client.get("/ready")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False
        assert body["checks"]["embedder"] is False
        assert "models" in body["checks"]["embedder_error"]

    def test_a_dimension_mismatch_is_not_ready_and_says_the_dimension(
        self, client, monkeypatch
    ):
        """1536-d queries against a 384-d index make every distance meaningless."""
        monkeypatch.setattr(
            "edgar_intel.providers.get_embedder", lambda: _embedder([[0.0] * 1536])
        )
        r = client.get("/ready")
        assert r.status_code == 503
        checks = r.json()["checks"]
        assert checks["embedder"] is False
        assert checks["embedder_dim"] == 1536
        assert "384" in checks["embedder_error"]

    def test_an_empty_vector_is_not_ready(self, client, monkeypatch):
        monkeypatch.setattr("edgar_intel.providers.get_embedder", lambda: _embedder([[]]))
        assert client.get("/ready").status_code == 503

    def test_an_empty_index_is_still_caught(self, client, monkeypatch):
        monkeypatch.setattr(
            "edgar_intel.providers.get_embedder", lambda: _embedder([[0.0] * 384])
        )
        monkeypatch.setattr(app_module.db, "query_one", lambda *a, **k: {"n": 0})
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["index"] is False

    def test_health_does_not_depend_on_any_of_it(self, client, monkeypatch):
        """Restarting the process because Postgres blinked makes an outage worse."""

        def boom(*a, **k):
            raise RuntimeError("database down")

        monkeypatch.setattr(app_module.db, "healthcheck", boom)
        monkeypatch.setattr("edgar_intel.providers.get_embedder", boom)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestThePaidEndpointIsGated:
    def test_no_key_configured_means_open(self, client, monkeypatch):
        """Local development and docker-compose must keep working unchanged."""
        monkeypatch.setenv("EDGAR_API_KEY", "")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()
        req = _fake_request()
        app_module.require_key(req)  # does not raise

    def test_a_configured_key_is_required(self, monkeypatch):
        monkeypatch.setenv("EDGAR_API_KEY", "s3cret")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()
        with pytest.raises(fastapi.HTTPException) as exc:
            app_module.require_key(_fake_request())
        assert exc.value.status_code == 401

    def test_the_right_key_passes(self, monkeypatch):
        monkeypatch.setenv("EDGAR_API_KEY", "s3cret")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()
        app_module.require_key(_fake_request(headers={"x-api-key": "s3cret"}))

    def test_a_wrong_key_fails(self, monkeypatch):
        monkeypatch.setenv("EDGAR_API_KEY", "s3cret")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()
        with pytest.raises(fastapi.HTTPException):
            app_module.require_key(_fake_request(headers={"x-api-key": "s3cre"}))

    def test_ask_returns_401_over_http_without_the_key(self, client, monkeypatch):
        monkeypatch.setenv("EDGAR_API_KEY", "s3cret")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()
        r = client.post("/ask", json={"question": "What was Apple FY2025 revenue?"})
        assert r.status_code == 401

    def test_search_stays_open_because_retrieval_is_free(self, client, monkeypatch):
        """A live URL must be demonstrable without handing out a key."""
        monkeypatch.setenv("EDGAR_API_KEY", "s3cret")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()

        class R:
            hits: list = []
            latency_ms = 3
            stage_latency_ms: dict = {}

        monkeypatch.setattr(app_module, "search", lambda *a, **k: R())
        r = client.post("/search", json={"query": "supplier concentration"})
        assert r.status_code == 200


class TestTheRateLimitOnTheSpendingPath:
    def test_requests_under_the_limit_pass(self, monkeypatch):
        _limit(monkeypatch, 3)
        app_module.reset_rate_limits()
        for i in range(3):
            app_module.enforce_rate_limit(_fake_request(), now=1000.0 + i)

    def test_the_limit_is_enforced(self, monkeypatch):
        _limit(monkeypatch, 3)
        app_module.reset_rate_limits()
        for i in range(3):
            app_module.enforce_rate_limit(_fake_request(), now=1000.0 + i)
        with pytest.raises(fastapi.HTTPException) as exc:
            app_module.enforce_rate_limit(_fake_request(), now=1003.0)
        assert exc.value.status_code == 429
        assert "3/min" in exc.value.detail

    def test_the_window_expires(self, monkeypatch):
        _limit(monkeypatch, 2)
        app_module.reset_rate_limits()
        app_module.enforce_rate_limit(_fake_request(), now=1000.0)
        app_module.enforce_rate_limit(_fake_request(), now=1001.0)
        with pytest.raises(fastapi.HTTPException):
            app_module.enforce_rate_limit(_fake_request(), now=1002.0)
        app_module.enforce_rate_limit(_fake_request(), now=1062.0)

    def test_the_budget_is_per_client_not_global(self, monkeypatch):
        """A global limit lets one client deny the endpoint to everyone."""
        _limit(monkeypatch, 1)
        app_module.reset_rate_limits()
        a = _fake_request(headers={"x-forwarded-for": "1.1.1.1"})
        b = _fake_request(headers={"x-forwarded-for": "2.2.2.2"})
        app_module.enforce_rate_limit(a, now=1000.0)
        app_module.enforce_rate_limit(b, now=1000.0)
        with pytest.raises(fastapi.HTTPException):
            app_module.enforce_rate_limit(a, now=1000.0)

    def test_zero_disables_the_limit(self, monkeypatch):
        _limit(monkeypatch, 0)
        app_module.reset_rate_limits()
        for i in range(50):
            app_module.enforce_rate_limit(_fake_request(), now=1000.0)

    def test_the_last_forwarded_hop_is_the_trusted_one(self):
        """The client can prepend to x-forwarded-for; the proxy appends."""
        req = _fake_request(headers={"x-forwarded-for": "9.9.9.9, 203.0.113.7"})
        assert app_module.client_ip(req) == "203.0.113.7"

    def test_no_forwarded_header_falls_back_to_the_socket(self):
        assert app_module.client_ip(_fake_request()) == "127.0.0.1"


class TestConfigReportsWhetherThePaidPathIsProtected:
    def test_it_reports_the_state_not_the_secret(self, client, monkeypatch):
        monkeypatch.setenv("EDGAR_API_KEY", "s3cret")
        monkeypatch.setenv("EDGAR_API_RATE_LIMIT_PER_MIN", "20")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()
        body = client.get("/config").json()
        assert body["ask_requires_api_key"] is True
        assert body["ask_rate_limit_per_min"] == 20
        assert "s3cret" not in str(body)

    def test_an_unprotected_instance_says_so(self, client, monkeypatch):
        monkeypatch.setenv("EDGAR_API_KEY", "")
        from edgar_intel.config import reset_settings_cache

        reset_settings_cache()
        assert client.get("/config").json()["ask_requires_api_key"] is False


class TestTheImageCanActuallyServeAQuery:
    """Structural guards on the build, because the defect was in the build file.

    These read the Dockerfile as text rather than building it. Building the image
    in CI would be the real check and takes minutes; this catches the specific
    regression -- someone trimming the extras to shrink the image -- in
    milliseconds.
    """

    @staticmethod
    def _dockerfile() -> str:
        from pathlib import Path

        return Path(__file__).resolve().parents[1].joinpath("Dockerfile").read_text()

    @staticmethod
    def _fly() -> str:
        from pathlib import Path

        return Path(__file__).resolve().parents[1].joinpath("fly.toml").read_text()

    def test_the_models_extra_is_installed(self):
        """Without it the embedder raises ImportError on every query."""
        assert "models" in self._dockerfile()

    def test_torch_comes_from_the_cpu_index(self):
        """The default wheel is the CUDA build: ~2.5 GB for a CPU-only service."""
        text = self._dockerfile()
        assert "download.pytorch.org/whl/cpu" in text
        cpu_torch = text.index("download.pytorch.org/whl/cpu")
        extras = text.index('pip install ".[serving,obs,models]"')
        assert cpu_torch < extras, "CPU torch must be pinned before the extras resolve it"

    def test_the_healthcheck_uses_readiness_not_liveness(self):
        """/health cannot detect a container that cannot embed."""
        assert "/ready" in self._dockerfile()

    def test_the_container_does_not_run_as_root(self):
        assert "USER app" in self._dockerfile()

    def test_the_machine_has_room_for_two_transformer_models(self):
        """512 MB OOM-kills partway through the first request."""
        assert 'memory = "1gb"' in self._fly() or 'memory = "2gb"' in self._fly()

    def test_no_secret_is_committed_in_the_fly_config(self):
        text = self._fly()
        assert "EDGAR_LLM_API_KEY =" not in text
        assert "EDGAR_DB_DSN =" not in text
        assert "EDGAR_API_KEY =" not in text
        assert "sk-" not in text

    def test_the_rate_limit_is_set_in_the_deployed_env(self):
        assert "EDGAR_API_RATE_LIMIT_PER_MIN" in self._fly()


# ------------------------------------------------------------------- helpers
def _limit(monkeypatch, n: int) -> None:
    monkeypatch.setenv("EDGAR_API_RATE_LIMIT_PER_MIN", str(n))
    from edgar_intel.config import reset_settings_cache

    reset_settings_cache()


def _fake_request(headers: dict[str, str] | None = None):
    from starlette.datastructures import Headers

    class Client:
        host = "127.0.0.1"

    class Req:
        def __init__(self) -> None:
            self.headers = Headers(headers or {})
            self.client = Client()

    return Req()
