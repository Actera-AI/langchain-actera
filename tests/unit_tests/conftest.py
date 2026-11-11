import pytest

from tests.unit_tests.fakes import FakeActera


@pytest.fixture
def fake() -> FakeActera:
    """An Actera where refunds need approval and deleting customers is blocked."""
    return FakeActera({"refund": "REQUIRE_APPROVAL", "delete_customer": "BLOCK"})


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ACTERA_API_KEY", "ACTERA_AGENT_ID", "ACTERA_BASE_URL", "ACTERA_MODE", "ACTERA_EVALUATION_FAILURE"):
        monkeypatch.delenv(name, raising=False)
