#!/usr/bin/env python3
"""Build docs/car_v5/car_v5_progress_report.html — a tier-by-tier "what changed vs the previous
version" report. Reuses the existing car_v5_rerun_report.html head (CDN 3Dmol + full CSS), its 3D
active-site viewer (embedded WT+Mg PDB), and its 3Dmol init script; rewrites the body with the
E1->E2->E4a progression + inline SVG charts."""
import re, pathlib

SRC = pathlib.Path("docs/car_v5/car_v5_rerun_report.html")
OUT = pathlib.Path("docs/car_v5/car_v5_progress_report.html")
lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
txt = "".join(lines)

def between(start_marker, end_marker, s=txt):
    i = s.index(start_marker); j = s.index(end_marker, i) + len(end_marker)
    return s[i:j]

# --- reused blocks from the built report ---
head = txt[:txt.index("</head>") + len("</head>")]
head = head.replace("<title>CAR V5 — 프로덕션 재현 · 진단 · 재실행 보고서</title>",
                    "<title>CAR V5 — 진행 경과 & 이전 버전 대비 변경점</title>")
vbar_and_viewer = between('<div class="vbar">', '</script></div>')   # controls + viewer div + PDB
legend = between('<div class="legend">', '</p>')                      # legend + structure caption
dmol_script = between('<script>\nfunction cssv', 'window.addEventListener(\'load\',vInit);\n</script>')

# ---------------------------------------------------------------- SVG charts (self-contained) ----
INK, MUT, TEAL, GOOD, RUST, AMB, BAD, LINE = "#1b1c1e","#6c6f74","#1f6f6b","#2f7d53","#b6532a","#b8860b","#c0392b","#e4e1d9"

def sx(a, lo=2.5, hi=6.5, x0=70, x1=850):   # distance-axis mapping
    return x0 + (a-lo)/(hi-lo)*(x1-x0)

chartA = f'''<svg width="100%" viewBox="0 0 900 176" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
<rect x="70" y="40" width="{sx(3.6)-70:.0f}" height="70" fill="{GOOD}" opacity="0.10"/>
<text x="{(70+sx(3.6))/2:.0f}" y="34" text-anchor="middle" font-size="12" fill="{GOOD}">반응 가능 구역 ≤3.6 Å</text>
<line x1="70" y1="110" x2="850" y2="110" stroke="{INK}" stroke-width="1.5"/>
{"".join(f'<line x1="{sx(t):.0f}" y1="106" x2="{sx(t):.0f}" y2="114" stroke="{MUT}"/><text x="{sx(t):.0f}" y="130" text-anchor="middle" font-size="11" fill="{MUT}">{t}</text>' for t in (2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5))}
<circle cx="{sx(2.789):.0f}" cy="110" r="7" fill="{GOOD}"/><text x="{sx(2.789):.0f}" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="{GOOD}">결정구조 목표</text><text x="{sx(2.789):.0f}" y="82" text-anchor="middle" font-size="12" fill="{GOOD}">2.79 Å (반응 자세)</text>
<circle cx="{sx(4.70):.0f}" cy="110" r="6" fill="{AMB}"/><text x="{sx(4.70):.0f}" y="156" text-anchor="middle" font-size="12" fill="{AMB}">E1 WT 최소 4.70</text>
<circle cx="{sx(5.12):.0f}" cy="110" r="6" fill="{MUT}"/><text x="{sx(5.12):.0f}" y="90" text-anchor="middle" font-size="11" fill="{MUT}">E1 WT 평균 5.12</text>
<circle cx="{sx(4.85):.0f}" cy="110" r="5" fill="{MUT}" opacity="0.6"/>
<line x1="{sx(2.789):.0f}" y1="110" x2="{sx(4.70):.0f}" y2="110" stroke="{BAD}" stroke-width="2" stroke-dasharray="5 3"/>
<text x="{(sx(2.789)+sx(4.70))/2:.0f}" y="156" text-anchor="middle" font-size="12" font-weight="700" fill="{BAD}">✗ MD가 못 메운 간극 (~1.9 Å)</text>
</svg>'''

def barsvg(items, maxv, unit, thresh=None, h=196, colfn=None):
    n=len(items); bw=90; gap=(760-n*bw)/(n+1); base=150
    out=[f'<svg width="100%" viewBox="0 0 900 {h}" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    out.append(f'<line x1="60" y1="{base}" x2="850" y2="{base}" stroke="{INK}" stroke-width="1.2"/>')
    if thresh is not None:
        ty=base-thresh/maxv*110
        out.append(f'<line x1="60" y1="{ty:.0f}" x2="850" y2="{ty:.0f}" stroke="{MUT}" stroke-dasharray="4 3"/><text x="856" y="{ty+4:.0f}" font-size="11" fill="{MUT}">{thresh}{unit}</text>')
    for i,(lab,val,sub) in enumerate(items):
        x=70+gap+i*(bw+gap); bh=val/maxv*110; y=base-bh
        col=colfn(val) if colfn else TEAL
        out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw}" height="{bh:.0f}" rx="5" fill="{col}"/>')
        out.append(f'<text x="{x+bw/2:.0f}" y="{y-8:.0f}" text-anchor="middle" font-size="14" font-weight="700" fill="{col}">{val}{unit}</text>')
        out.append(f'<text x="{x+bw/2:.0f}" y="{base+18:.0f}" text-anchor="middle" font-size="12" font-weight="650" fill="{INK}">{lab}</text>')
        if sub: out.append(f'<text x="{x+bw/2:.0f}" y="{base+34:.0f}" text-anchor="middle" font-size="11" fill="{MUT}">{sub}</text>')
    out.append('</svg>'); return "".join(out)

chartB = barsvg([("implicit GBSA",0.52,"WT 확산"),("explicit (E1)",1.00,"WT 유지")],1.15,"",
                colfn=lambda v:(GOOD if v>=0.9 else RUST))
chartC = barsvg([("WT",169,"control"),("lead",98,"3점"),("P438N",112,"단일"),
                 ("ctrl-A",131,"scalar"),("ctrl-B",95,"binding")],180,"°",
                thresh=150,colfn=lambda v:(GOOD if v>=150 else RUST))
chartD_bars = barsvg([("WT",8.45,"기준"),("lead",7.71,"3점 변이"),("P438N",6.94,"단일 변이")],9.5," ",
                colfn=lambda v:"#b9b6ad")
chartD = chartD_bars.replace("</svg>",
    f'<rect x="40" y="12" width="820" height="150" fill="{BAD}" opacity="0.06"/>'
    f'<text x="450" y="96" text-anchor="middle" font-size="30" font-weight="800" fill="{BAD}" opacity="0.55" transform="rotate(-8 450 96)">미수렴 · 해석 금지 (겹침 0.00)</text></svg>')

# ---------------------------------------------------------------- body ----
hero = '''<div class="hero">
  <h1>CAR V5 — 진행 경과 &amp; 이전 버전 대비 변경점</h1>
  <p class="sub">mechanism-configurable 효소 변이 triage 플랫폼. CAR(3-HP 아데닐화)를 acceptance test로, 검증 단계를 한 tier씩 올리며 <b>매번 무엇이 달라졌고 무엇을 알아냈는지</b> 정리했습니다. 활성 예측기가 아니라, 정직한 근거만 내는 플랫폼입니다.</p>
  <div class="meta"><span class="pill">실측 데이터 기반</span><span class="pill">Mg-일관</span><span class="pill">claim-safe</span><span class="pill">PR #22</span></div>
</div>'''

summary = f'''<div class="callout key"><span class="lbl">한눈에</span>
검증은 <b>implicit → corrected-Mg → E1 명시적 용매 → E2 결정구조 → E4a PMF</b> 순으로 올라갔습니다. 각 단계는 앞 단계의 <b>한 가지 약점</b>을 고쳤고, 마지막에 남은 벽은 <b>고정전하 힘장이 반응 거리(~2.8 Å)를 못 잡는다</b>는 것. E4a는 그 벽의 <b>높이(접근 비용)</b>를 재려는 단계입니다.</div>'''

# tier steps
def step(cls,n,title,changed,found,tag):
    return f'''<div class="step {cls}"><div class="n">{n}</div><div class="b"><h3>{title}</h3>
    <p><b>바뀐 점:</b> {changed}<br><b>알아낸 것:</b> {found}</p>{tag}</div></div>'''

steps = "".join([
 step("prob","1","implicit GBSA (focused 2 ns) — 이전 baseline",
   "최초 프로덕션 tier. GBSA 암시적 용매, 2 ns.",
   "<b>Mg 배치 버그</b>로 후보 MD가 Mg 없이 실행 → 결과 <b>교란(confounded)</b>. WT 공-기질(3-HP) 확산(잔류 0.52), O→P ~4.85 Å, 후보 구분 안 됨.",
   '<span class="tag t-bad">철회 (confounded)</span>'),
 step("fix","2","corrected-Mg smoke — 버그 수정",
   "<code>run_md_batches</code> 팬아웃에 <code>metal_requested</code> 전달(직렬만 됐고 배치 경로가 누락됐던 것). 후보도 Mg 획득.",
   "다중-GPU 배치에서도 Mg 다리가 실제로 삽입됨(구조 수준 MG 이온, OpenFF 아님). 기하 유효.",
   '<span class="tag t-good">기하 유효</span>'),
 step("fix","3","E1 — 명시적 용매(TIP3P + PME)",
   "암시적 GBSA → <b>명시적 물 ~95k + PME + Na/Cl</b>. Mg 일관.",
   "<b>WT 잔류율 0.52 → 1.00</b> — 확산 문제 해결, 처음으로 <b>의미 있는 WT baseline</b> 확보. 단 O→P는 여전히 ~4.7–5.1 Å, 각도 WT 169°/후보 95–131°.",
   '<span class="tag t-warn">CONDITIONAL</span>'),
 step("","4","E2 — 결정구조 기준(5MST)",
   "실측 아데닐화 활성 상태 결정구조를 <b>ground-truth</b>로 도입.",
   "결정구조 productive O→P = <b>2.79 Å / 119°</b>. E1 MD는 ≤3.6 Å 재진입 <b>0회</b>(prodOcc=0); WT는 <b>각도는 유지(169°)하나 거리는 못 감</b> → 원인은 solvent/reference가 아니라 <b>고정전하 힘장의 한계</b>.",
   '<span class="tag t-neu">기전 한계 규명</span>'),
 step("prob","5","E4a PMF — 우산 샘플링 (k=10, 첫 시도)",
   "<b>새 방법 도입:</b> O→P 거리 좌표에 조화 바이어스(우산) + WHAM으로 <b>접근 비용</b> 측정. 30창.",
   "창끼리 <b>겹침 0.00</b> → 미수렴. 프로그램이 스스로 <b>“해석 금지”</b> 판정(겉보기 순위는 있었으나 신뢰 불가). k=10 용수철이 너무 약함.",
   '<span class="tag t-bad">미수렴 · 재실행</span>'),
 step("fix","6","E4a PMF — 우산 샘플링 (k=60, 진행 중)",
   "<b>용수철 6배(k=60) + 창 24개</b>(촘촘). 바이어스가 국소 지형을 지배 → 창이 제자리 + 겹침.",
   "72창(WT+2후보) 스윕 진행 중. 수렴 시 <b>접근 비용</b> 해석; 평평하면 QM/MM-lite가 정직한 다음 tier.",
   '<span class="tag t-warn">진행 중</span>'),
])

session_code = '''<h2>이번 세션에서 바뀐 코드 (PR #22)</h2>
<div class="scroll"><table>
<thead><tr><th>영역</th><th>이전</th><th>지금</th></tr></thead><tbody>
<tr><td>E4a 리포트</td><td>접근비용만 보고</td><td><b>접근 장벽 ↔ NAC 점유 분리</b> (거리 바이어스라 짧은 거리 ≠ 일직선; 각도≥150° 비율 별도), WHAM <b>겹침/수렴 QC</b>, strict schema + claim ceiling</td></tr>
<tr><td>우산 샘플링 실행</td><td>분석기만 있고 <b>샘플 생성기 없음</b></td><td>s10 <b>ENV-gated 스윕</b>(purge-safe) + <code>umbrella_samples.json</code> 저장 + 드라이버가 이를 읽음</td></tr>
<tr><td>친핵체 종류</td><td>리간드 원자만(CAR·FDH)</td><td><b>단백질 잔기 친핵체</b>(serine hydrolase Ser Oγ) 지원 — 일반 기능, TEM-1에 배선</td></tr>
</tbody></table></div>
<p><small>테스트: 798 통과 · 기존 실패 9(전부 사전 존재) · 신규 실패 0.</small></p>'''

charts = f'''<h2>핵심 데이터 (실측)</h2>
<div class="card"><h3>① 반응 거리: 결정구조 목표 vs MD 도달값</h3>{chartA}
<p><small>결정구조 목표는 2.79 Å이지만, 명시적 MD도 WT를 4.70 Å까지밖에 못 가져감 — E2가 규명한 <b>힘장 한계</b>.</small></p></div>
<div class="grid2" style="margin-top:16px">
<div class="card"><h3>② WT 잔류율 — E1이 고친 것</h3>{chartB}
<p><small>명시적 용매가 공-기질 확산을 해결(0.52→1.00).</small></p></div>
<div class="card"><h3>③ 각도(E1) — WT만 일직선</h3>{chartC}
<p><small>WT만 in-line(169°) 유지. 각도 차이는 진짜지만, 거리는 아무도 반응 자세(≤3.6 Å) 아님.</small></p></div></div>
<div class="card" style="margin-top:16px"><h3>④ E4a k=10 접근비용 — 왜 못 믿나</h3>{chartD}
<p><small>겉보기 순위(WT 8.45 &gt; lead 7.71 &gt; P438N 6.94)는 있으나 창이 안 겹쳐(0.00) <b>미수렴</b> → k=60으로 재실행 중.</small></p></div>'''

structure_head = '''<h2>WT 활성부위 3D 구조 — Mg²⁺ 다리 (수정된 상태)</h2>
<p>이전 focused run에서 <b>후보엔 빠졌던</b> Mg²⁺가, 수정 후 3-HP 카복실기 산소와 ATP 인산기 사이에 다리를 놓습니다. Mg가 3-HP 친핵성 산소를 ~1.9 Å로 배위. 노란 점선 = 반응 축 O→Pα(거리 표시). 드래그 회전 · 스크롤 확대.</p>'''

claim = '''<h2>Claim discipline (계속 유지)</h2>
<div class="callout warn"><span class="lbl">할 수 있는 말 / 없는 말</span>
<b>가능:</b> “스크리닝 수준에서 A가 반응 거리에 더 싸게 접근한다.” &nbsp;
<b>금지:</b> “더 좋은 효소 / 활성 증가 / kcat 상승 / 활성화 장벽 감소 / 검증된 lead.”
접근 비용은 <b>고정전하 힘장의 거리축 자유에너지</b>일 뿐 활성화 에너지가 아니며(그건 QM/MM 필요), 이전 focused 랭킹은 <b>철회 상태</b>입니다.</div>'''

foot = '''<div class="foot">
  EvoLiEZ · CAR V5 진행 경과 보고 · <code>feat/car-v5-reviewer2-refinements</code> · PR #22<br>
  수치 출처: 서버 provenance(<code>md/*/analysis.json</code>, <code>docs/car_v5/e1/e2/e3 tables</code>, <code>umbrella_samples.json</code>).
  대용량 trajectory는 서버/GDrive 보관. 후보 순위·활성 주장 없음.
</div>'''

body = (hero + '<p class="lead">단계별로 “이전 대비 무엇이 달라졌는지 + 무엇을 알아냈는지”를 정리합니다.</p>'
        + summary
        + '<h2>검증 단계의 진화 (무엇이 달라졌나)</h2>' + steps
        + session_code + charts
        + structure_head + vbar_and_viewer + legend
        + claim + foot)

html = (head + '\n<body>\n<div class="wrap">\n' + body + '\n</div>\n<script>\n'
        + dmol_script + "window.addEventListener('load',vInit);\n</script>\n</body>\n</html>\n")
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, "(%d KB)" % (len(html)//1024))
