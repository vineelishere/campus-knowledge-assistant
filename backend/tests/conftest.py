"""Shared test setup.

Tracing is disabled before any app module is imported so tests never attempt Langfuse
network calls because of whatever happens to be in the developer's .env.
"""
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ.setdefault("ENVIRONMENT", "test")
