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
    poll_tries: int = 180,
) -> Optional[List[Tuple[str, str]]]:
    """Return [(id, aligned_seq)] with the query first, or None on any failure.

    Implements the ColabFold MMseqs2 API: POST a FASTA query to /ticket/msa ->
    {"id","status"}; poll /ticket/<id> until COMPLETE; download the result
    (a gzipped TAR of .a3m files) from /result/download/<id> and parse the
    largest a3m. (The previous version used the whole ticket JSON as the id ->
    HTTP 400, and read the tar.gz as raw text.)"""
    try:
        import json
        import urllib.parse
        import urllib.request
    except Exception:  # pragma: no cover
        return None

    workdir.mkdir(parents=True, exist_ok=True)

    def _get_json(url):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())

    try:
        query = f">query\n{sequence}\n"
        data = urllib.parse.urlencode({"q": query, "mode": "all"}).encode()
        req = urllib.request.Request(f"{api_base}/ticket/msa", data=data)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            sub = json.loads(resp.read().decode())
        tid = sub.get("id")
        status = sub.get("status", "")
        if not tid:
            log.warning("remote MSA: no ticket id in %s", str(sub)[:120])
            return None
        log.info("remote MSA ticket %s (%s)", tid, status)
        for _ in range(poll_tries):
            if status == "COMPLETE":
                break
            if status in ("ERROR", "MAINTENANCE", "UNKNOWN", "RATELIMIT"):
                log.warning("remote MSA status=%s", status)
                return None
            time.sleep(5)
            status = _get_json(f"{api_base}/ticket/{tid}").get("status", "")
        if status != "COMPLETE":
            log.warning("remote MSA did not COMPLETE (last status=%s)", status)
            return None
        with urllib.request.urlopen(
            f"{api_base}/result/download/{tid}", timeout=timeout
        ) as r:
            payload = r.read()
        a3m_text = _extract_a3m(payload)
        if not a3m_text:
            log.warning("remote MSA: no a3m in the downloaded result")
            return None
        a3m = workdir / "remote.a3m"
        a3m.write_text(a3m_text)
        return _read_a3m(a3m)
    except Exception as exc:
        log.warning("remote MSA failed (%s); caller will fall back", exc)
        return None


def _extract_a3m(payload: bytes) -> Optional[str]:
    """The ColabFold result is a gzipped TAR of .a3m files (uniref.a3m, bfd...).
    Return the largest .a3m's text (most homologs); tolerate a raw a3m too."""
    import io
    import tarfile

    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            best = ""
            for m in tar.getmembers():
                if not m.name.endswith(".a3m"):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                txt = f.read().decode("utf-8", "ignore")
                if len(txt) > len(best):
                    best = txt
            return best or None
    except (tarfile.TarError, OSError):
        try:                                   # maybe it's already a raw a3m
            txt = payload.decode("utf-8", "ignore")
            return txt if txt.lstrip().startswith(">") else None
        except Exception:
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
