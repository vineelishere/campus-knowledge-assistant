"""Thin wrapper so the app runs with or without Langfuse configured."""
try:
    from langfuse import observe  # langfuse >= 3
except Exception:  # pragma: no cover - langfuse not installed
    def observe(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn):
            return fn

        return deco

try:
    from langfuse import get_client
except Exception:  # pragma: no cover
    def get_client():
        return None


def update_trace(**kwargs) -> None:
    """Attach user/session/metadata to the current trace; never let tracing break a request."""
    try:
        client = get_client()
        if client is not None:
            client.update_current_trace(**kwargs)
    except Exception:
        pass


def update_generation(**kwargs) -> None:
    """Attach model name / token usage to the current generation span (shown in Langfuse)."""
    try:
        client = get_client()
        if client is not None:
            client.update_current_generation(**kwargs)
    except Exception:
        pass
