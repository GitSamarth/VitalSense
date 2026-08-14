"""Centralised secret / config resolution for VitalSense.

Priority order:
  1. Environment variable (e.g. set via Docker -e, a .env loader, or CI secrets)
  2. Streamlit st.secrets (local development via .streamlit/secrets.toml)

This lets the app run identically in both environments without any code changes
at call sites — Docker passes secrets as env vars, local dev uses secrets.toml.
"""
import os
import streamlit as st


def get_secret(key: str) -> str:
    """Return the value for *key* from env vars (priority) or st.secrets (fallback).

    Returns an empty string if the key is absent from both sources, so callers
    can do a simple truthiness check (``if get_secret("KEY")``) without catching
    KeyError.
    """
    # os.environ takes priority — set by Docker -e, CI/CD, or a local .env loader
    env_val = os.environ.get(key)
    if env_val:
        return env_val
    # Fall back to Streamlit secrets for local dev (secrets.toml)
    return st.secrets.get(key, "")
