from types import SimpleNamespace

from document_agent.api.app import _is_request_authorized


def _request(path: str, headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(url=SimpleNamespace(path=path), headers=headers or {})


def test_api_key_auth_disabled_by_default() -> None:
    settings = SimpleNamespace(api_key=None, api_key_header="X-API-Key")
    assert _is_request_authorized(_request("/v1/jobs"), settings) is True


def test_api_key_auth_accepts_configured_header() -> None:
    settings = SimpleNamespace(api_key="secret", api_key_header="X-API-Key")
    assert _is_request_authorized(_request("/v1/jobs", {"X-API-Key": "secret"}), settings) is True


def test_api_key_auth_rejects_missing_key() -> None:
    settings = SimpleNamespace(api_key="secret", api_key_header="X-API-Key")
    assert _is_request_authorized(_request("/v1/jobs"), settings) is False


def test_api_key_auth_leaves_health_checks_open() -> None:
    settings = SimpleNamespace(api_key="secret", api_key_header="X-API-Key")
    assert _is_request_authorized(_request("/readyz"), settings) is True
