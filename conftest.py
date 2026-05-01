"""Root conftest — sets env vars required for module-level app import.

AnalyticsAgentApp() is instantiated at module level in src/app.py so that
uvicorn can import the `app` symbol without further setup. The SDK 0.6.0
BaseAgentApp.__init__ calls config_model.load(), which needs ENVIRONMENT and
CONFIG_DIR (pointing at the local config/ tree) and INTERNAL_API_KEY (the
only ${VAR} reference in config/default.yaml).

These env vars are set here — before any test module is collected — so that
the import of src.app succeeds in the test process without requiring a
.env file or a running Docker stack.
"""

import os
from pathlib import Path

# Resolve to the project root regardless of where pytest is invoked from.
_PROJECT_ROOT = Path(__file__).parent

os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("CONFIG_DIR", str(_PROJECT_ROOT / "config"))
os.environ.setdefault("INTERNAL_API_KEY", "test-key")
