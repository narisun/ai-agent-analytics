"""HTTP transport layer.

Each module exports an APIRouter that delegates to a service. Routers
are registered by ``create_app(deps)`` in ``src/app.py``. Routes do
NOT call platform-sdk directly — that's the domain/service layer's
job.
"""
