"""
Shared UI configuration — single source of truth for all components.
"""
import os

API_BASE = os.environ.get("NEURAL_SEARCH_API_URL", "http://localhost:8000")
