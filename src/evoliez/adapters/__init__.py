"""External-tool adapters.

Every heavy tool is wrapped with two backends:

* ``real`` - shells out to the actual binary on the GPU server.
* ``mock`` - deterministic synthetic output so the whole pipeline runs on a
  laptop / in CI with identical data flow, schema and scoring.

Backend is chosen per stage from the config (see ``Config.backend_for``).
"""
