"""Optional hosted MSA (spec 9.1 "use MSA server").

When local sequence databases are unavailable (the dev box has tiny disk; the
server's root is full), an MSA can be fetched from a hosted MMseqs2 service
instead of downloading UniRef/BFD. Network-optional: any failure raises and the
caller falls back to the identity/synthetic alignment.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

from evoliez.logging_utils import get_logger

log = get_logger("evoliez.remote_msa")

DEFAULT_API = "https://api.colabfold.com"


def fetch_msa(
    sequence: str,
    workdir: Path,
    *,
    api_base: str = DEFAULT_API,
    timeout: float = 120.0,
) -> Optional[List[Tuple[str, str]]]:
    """Return [(id, aligned_seq)] with the query first, or None on any failure."""
    try:
        import urllib.parse
        import urllib.request
    except Exception:  # pragma: no cover
        return None

    workdir.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.parse.urlencode(
            {"q": sequence, "mode": "all"}
        ).encode()
        req = urllib.request.Request(f"{api_base}/ticket/msa", data=data)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ticket = resp.read().decode()
        log.info("submitted MSA ticket: %s", ticket[:80])
        # Poll for completion, then download the a3m.
        for _ in range(60):
            time.sleep(5)
            with urllib.request.urlopen(
                f"{api_base}/ticket/{ticket}", timeout=timeout
            ) as r:
                status = r.read().decode()
            if "COMPLETE" in status:
                break
            if "ERROR" in status:
                return None
        a3m = workdir / "remote.a3m"
        with urllib.request.urlopen(
            f"{api_base}/result/download/{ticket}", timeout=timeout
        ) as r:
            a3m.write_bytes(r.read())
        return _read_a3m(a3m)
    except Exception as exc:
        log.warning("remote MSA failed (%s); caller will fall back", exc)
        return None


def cached_fetch_msa(
    sequence: str, msa_dir: Path, *, api_base: str = DEFAULT_API,
    timeout: float = 120.0,
) -> Optional[List[Tuple[str, str]]]:
    """``fetch_msa`` but reuse a previously-downloaded ``remote.a3m`` in
    ``msa_dir`` so s02 (homolog extraction) and s03 (the alignment) share ONE
    ColabFold request instead of querying the API twice."""
    cached = msa_dir / "remote.a3m"
    if cached.exists() and cached.stat().st_size > 0:
        try:
            msa = _read_a3m(cached)
            if msa:
                log.info("reusing cached remote MSA (%d sequences)", len(msa))
                return msa
        except OSError:
            pass
    return fetch_msa(sequence, msa_dir, api_base=api_base, timeout=timeout)


def _read_a3m(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    cid, buf = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cid is not None:
                out.append((cid, "".join(buf)))
            cid, buf = line[1:].strip().split()[0], []
        else:
            # a3m: drop lowercase insertions to recover aligned columns
            buf.append("".join(c for c in line.strip() if not c.islower()))
    if cid is not None:
        out.append((cid, "".join(buf)))
    return out
