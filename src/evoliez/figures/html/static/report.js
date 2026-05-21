/* EvoLiEZ HTML report runtime.
 *
 * Vanilla JS, no modules, no jQuery.  Wrapped in an IIFE so it can be
 * inlined inside <script> in the report without leaking globals.
 *
 * Provides:
 *   1. IntersectionObserver highlighting the active TOC entry on scroll.
 *   2. Sortable / filterable / paginated <table class="datatable"> blocks.
 *
 * The 3Dmol viewer scripts are emitted per-viewer by the template and run
 * independently of this file.
 */

(function () {
  "use strict";

  var PAGE_SIZE = 25;

  // ---- TOC active-section highlight ----------------------------------
  function initTocSpy() {
    var links = document.querySelectorAll(".toc a[href^='#']");
    if (!links.length || typeof IntersectionObserver === "undefined") return;
    var byId = {};
    links.forEach(function (a) {
      var id = a.getAttribute("href").slice(1);
      byId[id] = a;
    });
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          var link = byId[entry.target.id];
          if (!link) return;
          if (entry.isIntersecting) {
            links.forEach(function (l) {
              l.classList.remove("toc-active");
            });
            link.classList.add("toc-active");
          }
        });
      },
      { rootMargin: "-30% 0px -60% 0px", threshold: 0 }
    );
    Object.keys(byId).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  }

  // ---- Datatable: sort + filter + paginate ---------------------------
  function cellValue(row, idx) {
    var c = row.children[idx];
    return c ? c.textContent.trim() : "";
  }

  function compareFactory(idx, asc) {
    return function (a, b) {
      var av = cellValue(a, idx);
      var bv = cellValue(b, idx);
      var an = parseFloat(av);
      var bn = parseFloat(bv);
      var bothNum =
        !isNaN(an) && !isNaN(bn) && /^-?\d/.test(av) && /^-?\d/.test(bv);
      var cmp;
      if (bothNum) cmp = an - bn;
      else cmp = av.localeCompare(bv);
      return asc ? cmp : -cmp;
    };
  }

  function applyPagination(state) {
    var rows = state.visibleRows;
    var pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    if (state.page >= pageCount) state.page = pageCount - 1;
    if (state.page < 0) state.page = 0;
    state.allRows.forEach(function (r) {
      r.classList.add("hidden");
    });
    var start = state.page * PAGE_SIZE;
    rows.slice(start, start + PAGE_SIZE).forEach(function (r) {
      r.classList.remove("hidden");
    });
    if (state.pager) {
      state.pager.info.textContent =
        "Page " + (state.page + 1) + " / " + pageCount;
      state.pager.prev.disabled = state.page === 0;
      state.pager.next.disabled = state.page >= pageCount - 1;
    }
    if (state.countEl) {
      state.countEl.textContent =
        rows.length + " / " + state.allRows.length + " rows";
    }
  }

  function applyFilter(state, query) {
    var q = query.trim().toLowerCase();
    if (!q) {
      state.visibleRows = state.allRows.slice();
    } else {
      state.visibleRows = state.allRows.filter(function (row) {
        return row.textContent.toLowerCase().indexOf(q) !== -1;
      });
    }
    state.page = 0;
    applyPagination(state);
  }

  function applySort(state, colIdx, th) {
    var current = th.getAttribute("aria-sort");
    var asc = current !== "ascending";
    state.headers.forEach(function (h) {
      h.removeAttribute("aria-sort");
    });
    th.setAttribute("aria-sort", asc ? "ascending" : "descending");
    state.allRows.sort(compareFactory(colIdx, asc));
    var tbody = state.table.tBodies[0];
    state.allRows.forEach(function (r) {
      tbody.appendChild(r);
    });
    state.visibleRows.sort(compareFactory(colIdx, asc));
    applyPagination(state);
  }

  function initDatatable(wrap) {
    var table = wrap.querySelector("table.datatable");
    if (!table || !table.tBodies[0]) return;
    var state = {
      table: table,
      headers: Array.prototype.slice.call(table.tHead.rows[0].cells),
      allRows: Array.prototype.slice.call(table.tBodies[0].rows),
      visibleRows: [],
      page: 0,
      pager: null,
      countEl: wrap.querySelector(".datatable-count"),
    };
    state.visibleRows = state.allRows.slice();

    state.headers.forEach(function (th, idx) {
      th.addEventListener("click", function () {
        applySort(state, idx, th);
      });
    });

    var search = wrap.querySelector(".datatable-search");
    if (search) {
      search.addEventListener("input", function () {
        applyFilter(state, search.value);
      });
    }

    if (state.allRows.length > PAGE_SIZE) {
      var pager = document.createElement("div");
      pager.className = "datatable-pager";
      var prev = document.createElement("button");
      prev.type = "button";
      prev.textContent = "Prev";
      var info = document.createElement("span");
      info.className = "datatable-pager-info";
      var next = document.createElement("button");
      next.type = "button";
      next.textContent = "Next";
      pager.appendChild(prev);
      pager.appendChild(info);
      pager.appendChild(next);
      wrap.appendChild(pager);
      state.pager = { prev: prev, next: next, info: info };
      prev.addEventListener("click", function () {
        state.page -= 1;
        applyPagination(state);
      });
      next.addEventListener("click", function () {
        state.page += 1;
        applyPagination(state);
      });
    }

    applyPagination(state);
  }

  function initDatatables() {
    var wraps = document.querySelectorAll(".datatable-wrap");
    wraps.forEach(function (w) {
      initDatatable(w);
    });
  }

  // ---- Smooth-scroll fallback for browsers without CSS smooth-scroll
  function initTocClicks() {
    var links = document.querySelectorAll(".toc a[href^='#']");
    links.forEach(function (a) {
      a.addEventListener("click", function (e) {
        var id = a.getAttribute("href").slice(1);
        var target = document.getElementById(id);
        if (!target) return;
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        history.replaceState(null, "", "#" + id);
      });
    });
  }

  function init() {
    initTocSpy();
    initTocClicks();
    initDatatables();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
