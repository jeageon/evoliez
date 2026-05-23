#!/usr/bin/env bash
# scripts/server_cheap_run_psefdh.sh — DEPRECATED THIN WRAPPER
#
# The implementation moved to scripts/run_server_cheap.sh, which is a
# single self-bootstrapping script (auto-activates the conda env,
# git-pulls if behind, runs editable install when modules are missing,
# fetches FASTA when absent, runs doctor + pipeline + bench-summary,
# prints a clear PASS/FAIL). The new script also handles all 5 enzyme
# cards via `ENZYME=fdh|xr|tem1|bgl3|p450`.
#
# This wrapper exists so existing automation / runbooks referencing the
# old script name continue to work. New work should call the target
# script directly:
#
#     bash scripts/run_server_cheap.sh                  # PseFDH (default)
#     ENZYME=xr bash scripts/run_server_cheap.sh        # XR
#
# Anything you pass here (--dry-run, CFG=..., BENCH=..., NAME=...) is
# forwarded verbatim.

set -euo pipefail
exec bash "$(dirname "$0")/run_server_cheap.sh" "$@"
