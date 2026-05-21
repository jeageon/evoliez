# Vendored JavaScript libraries

These files are bundled with the EvoLiEZ report so the output HTML works
offline (no network at view time), opens from `file://` URLs, and remains
viewable even if the upstream CDN goes away.

## 3Dmol.js

- **File:** `3Dmol-min.js`
- **Version:** 2.x (build downloaded from upstream stable URL)
- **Upstream:** <https://3dmol.csb.pitt.edu/>
- **Source URL:** <https://3Dmol.csb.pitt.edu/build/3Dmol-min.js>
- **License:** BSD 3-Clause
- **Repository:** <https://github.com/3dmol/3Dmol.js>
- **Citation:**
  Rego, N. & Koes, D. (2015). 3Dmol.js: molecular visualization with WebGL.
  *Bioinformatics*, 31(8), 1322-1324.
  doi:[10.1093/bioinformatics/btu829](https://doi.org/10.1093/bioinformatics/btu829)

### Why vendor it?

The report is meant to be portable: copy a single directory to a
collaborator's laptop, zip it for paper supplementary, or unzip it on an
air-gapped server.  External CDN references would break all of those use
cases.

### Refreshing

```bash
curl -L -o src/evoliez/figures/html/static/vendor/3Dmol-min.js \
  https://3Dmol.csb.pitt.edu/build/3Dmol-min.js
```

Sanity-check after refresh: the file should be roughly 0.5 MB and start
with a `/*! For license information ... */` banner.  Open any generated
report's 3D viewer and confirm the scene renders.

### License attribution

A copy of the BSD 3-Clause text ships in the file's own header comment
(see `3Dmol-min.js.LICENSE.txt` upstream).  Redistribution of EvoLiEZ
reports must preserve that notice.
