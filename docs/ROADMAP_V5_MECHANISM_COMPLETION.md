# ROADMAP V5 — Mechanism Completion & Catalytic-Geometry Validity

> **한 줄 요약**: v5는 *새 배관 공사가 아니라 완성(completion)*이다. v3/v4 merge-gate가 네 개
> "framework blocker"의 거친 버전을 이미 해결했으므로(multi-lane union은 s08/s08b/s09 경계에
> 이미 있음, ClaimGuard는 `page()`에 이미 연결됨, `Config.mechanism`은 이미 first-class
> field, s04x는 이미 조건부 삽입됨), 남은 일은 **(A) CAR 동적 NAC를 물리적으로 유효하게 만들기**와
> **(B) 이미 만들어져 있으나 소비되지 않는 mechanism/claim/evidence 배관을 실제로 연결하기**다.

- **상태**: DESIGN (2026-07-04, review round 2 반영). 근거: `feat/server-hardening` 브랜치 대상
  8-auditor file:line 검증 workflow(`v5-plan-groundtruth`, run `wf_9afb916c-d2b`) + 두 전문가 리뷰
  (v4 구현 + SrCAR→3-HP production run) + **round-2 git-ref 정정**(§0.5).
- **⚠️ round-2 정정(2026-07-04)**: §0의 "s08 already_fixed"는 **오판이었다**. multi-lane union은
  로컬 미푸시 커밋 `9ddf63f`에만 있고(그마저도 minimal), `origin/main`·`origin/feat/server-hardening`
  둘 다 여전히 순수 top-cut이다. **GitHub에서 보이는 코드 기준 s08은 아직 blocker다.** §0.5 참조.
- **착지 완료(2026-07-04, step 2 ClaimGuard 대부분 + step 3 시작)**:
  - **V5-5**: `recommendation()` geometry-gate(bare "strong candidate" 제거→"priority screening
    candidate"/"structural/binding candidate") · `is_paper_grade` verdict 정정(이중계상 제거) ·
    `claim_guard._PATTERNS`에 `UNQUALIFIED_STRENGTH` 카테고리(strong candidate·catalytic lead·
    paper-grade·confirmed productive; 절대 unlock 안 됨) · report copy 정정(paper_report "catalytic
    lead"→"top-ranked lead", paper_report_v2 "paper-grade"→"anchored-evaluated", md_movie, +docstrings).
    `page()` strict 강제 확인. **남음**: `io/_provenance_card.py`(real provenance) + 8개 자체-template
    report 전역 강제(V5-6).
  - **V5-3 부분**: `ReactionState.cofactor_state` 필드 추가(ATP 등 non-redox cofactor; CAR config
    `cofactor_state: ATP` 언블록). **purge-safe 검증됨** — 기존 config는 `mechanism=None`이라 dump에
    ReactionState 미포함 → config_sha1 불변.
  - **⚠️ purge-trap 발견**: `config_sha1 = sha1(config.model_dump(mode="json"))`(context.py:167)는
    **모든 defaulted field 포함** → **신규 top-level Config field는 기존 run 전부 purge**. 따라서
    `strict_mechanism_required`는 config field가 아니라 **ENV var `EVOLIEZ_STRICT_MECHANISM`**로(기존
    `EVOLIEZ_STRICT_CLAIMS` 패턴). §0.5.3 결정 4 수정.
  - **V5 step 3 완료**: `mechanism/mode.py`(purge-safe) — `enforce_mechanism_policy`(legacy banner,
    `EVOLIEZ_STRICT_MECHANISM`=1이면 mechanism 필수), `Pipeline.run` 시작에 배선. `EVOLIEZ_STRICT_MECHANISM`은
    ENV(config field 아님 — purge-safe).
  - **V5 step 4 완료**: `resolve_geometry_source()`(mechanism>legacy>none) — s10가 `geometry_source`를
    provenance(meta + md_candidates.json)에 기록. mechanism 소비는 `mechanism_spec_pending`(V5-3/V5-6까지
    정직하게 hook). `tests/test_v5_mechanism_mode.py`.
  - 검증: `tests/test_v5_claim_integrity.py` + `test_v5_mechanism_mode.py` + claim/evidence/mechanism/
    calibration suite green(**66 pass**). page() strict 강제 재확인.
  - **✅ RESUME 버그 FIXED**: 원인 = WIP-추가된 `s07.load()`가 `generated_candidates.json`에서 후보를
    id/mutations/generator만 복원하고 `cand.details["msa_permissiveness"]`(s08이 읽는 feature)를 DROP →
    resume 시 ml_score/final_score/ranked-set 조용히 발산(전형적 resume-load-parity trap). FIX: `s07.load()`가
    `position_features`(s03.load()가 먼저 복원)에서 msa_permissiveness를 run()과 동일 재계산. resume 8/8, 후보
    집합 bit-identical.
  - **✅ STEP 5 (s08 union + plate)**: `select_multi_lane`에 s08-COMPUTABLE lane 추가
    (`evolutionary_high`=msa_permissiveness, `ligand_competence_high`=interaction_gain) — inert한
    stability/geometry config slot로 크기 지정(purge-safe, 신규 Config field 없음); s08 low_ml_control 비면
    fail-loud; `lane_allocator.allocate()`를 s11에 ADDITIVE 96-well plate artifact로 배선
    (`plate_allocation.json`, uncalibrated flag) — ranking driver 아님(#6/#7로 flip 연기). `test_v5_s08_lanes.py`.
    slot 재사용은 s08 provenance(`s08_lane_quota_mapping`)에 명시 기록 + config 주석. **Migration plan**:
    후속 PR에서 `from_evolutionary_high`/`from_ligand_high`를 별도 Config field로 분리(현재는 purge-safe
    위해 stability/geometry slot 재사용).
  - **✅ STEP 6 core (CAR science, 로컬)**: **V5-1** O→P angle — `nac.py`가 leaving-O를 topologically
    (`identify_leaving`: acceptor-P의 bridging O→2nd P) 해석해 `transfer_is_h:false`에서 진짜
    O_nuc–Pα–O_leaving 각 계산(전엔 degenerate NaN→전 프레임 occupancy 0); engine이 leaving atom을 `nac_idx`에
    append(4-row subframe auto-detect). **V5-3** `adenylation_phosphoryl_transfer` 템플릿(Pα, Pγ stub과 구분) +
    CAR config `mechanism:` 선언(→ mechanism_spec mode). FDH hydride 경로 byte-identical.
    `test_v5_nac_angle.py`, `test_v5_adenylation_template.py`. **V5-2 (Mg²⁺ OpenMM)은 SERVER-gated.**
  - **✅ STEP 9 core (generality proof, V5-C)**: `test_v5_generality.py` — 4개 real-geometry 템플릿
    (hydride·acyl·glycosidic·adenylation) 모두 config-only 선언 가능·geometry emit·required reaction_state
    hard-gate·단일 GeometryTerm schema. (FDH/TEM-1/glycosidase E2E smoke는 data-gated.)
  - **✅ V5-2 core (Mg²⁺ placement, 로컬)**: `md/metal_placement.py` — `bridging_metal_position`
    (carboxylate O + phosphate O 사이 ~2.1 Å bridging 위치, **DETERMINISTIC**: WT/mutant 동일 배치 →
    유효 ΔNAC; reference로 perpendicular side 결정) + `insert_mg_into_pdb`(구조-레벨 MG HETATM →
    Amber ion params가 처리). `test_v5_metal_placement.py`. **engine 주입 call-site + Amber MG 검증은
    server-only** (placement 코어는 준비됨).
  - **✅ STEP 9 config-templates**: `configs/templates/mechanisms/{adenylation_phosphoryl_transfer,
    glycosidic_bond_cleavage}.yaml` 추가 → 4개 real-geometry 템플릿 모두 config-template surface 보유.
    `test_v5_mechanism_config_templates.py`가 YAML↔code registry 일관성 강제(drift 방지).
  - **Suite health**: V5 tests 전부 green(**45**, rdkit 1 skip); 남은 9개 suite 실패는 ALL pre-existing
    (recoverable stash로 확인 — my-edit 전에도 동일 실패: rdkit-missing·boltz CIF/PDB parser WIP·real
    dry-run·metalloenzyme config/fasta drift·reuse-dir-purge). Net: resume +fixed, 신규 실패 0.
  - **남은 것(전부 server/data-gated)**: V5-2 engine 주입(OpenMM), step 7 CAR Mg²⁺ rerun(GPU), step 8
    EvidenceCard ranking flip(flag-gated, rerun 후), step 9 FDH/TEM-1/glycosidase E2E smoke(target fasta 필요).
- **선행 문서**: `ROADMAP_V3_MECHANISM_TRIAGE.md`, `ROADMAP_V4_GUIDED_ENSEMBLE.md`,
  `ROADMAP_V3_V4_MERGE_AND_COMPLETION.md`.
- 관련 메모리: car-srcar-3hp-run, s10-nac-catalytic, anchored-validation-redesign,
  v3-completion-merge-gates, evoliez-v4-implementation, resume-load-parity.

---

## 0. Reframing — 전문가 blocker vs. 코드 검증 진실

전문가 리뷰의 큰 가치는 **방향**이 맞다는 점이다. 그러나 리뷰가 지목한 네 개 framework blocker의
"literal 주장"은 대부분 2026-07-02 merge-gate 이후 코드와 어긋난다. 아래는 auditor가 file:line
증거로 확인한 판정이다. **원문 주장이 아니라 이 판정 표가 v5의 출발점**이다.

| 영역 | 전문가 주장 | 검증 판정 | 실제 진실 (file:line) |
|---|---|---|---|
| **s08 multi-lane** | s08이 여전히 `ml_score` top-N으로 자름 → 저-ML 후보 사멸 | **전문가 맞음 (round-2 정정)** | **GitHub 기준 s08은 순수 top-cut**(`origin/main`, `origin/feat/server-hardening` 둘 다 `top=candidates[:top_for_redocking]`). union은 로컬 미푸시 커밋 `9ddf63f`에만 있고 그것도 `ml_high+diversity+low_ml_control`만(`s08:249-254`, `from_stability_high=0/from_geometry_high=0`). 요구 lane(mechanism_geometry/ligand_role/structural_evolutionary/uncertainty_probe/deconvolution/known_controls) **전무**. `lane_allocator.py`는 이들 lane을 **완비했으나 unwired**(grep `allocate(` in stages/ = empty). §0.5·V5-7 참조. |
| **ClaimGuard 전역 강제** | report writer에 강제 안 됨 | **partially_correct** | `_report_kit.page()`에 이미 연결(`_claimguard_gate`) + markdown 경로(`report.py`). 그러나 (1) 14개 HTML writer 중 **6개만** `page()` 경유(8개 자체 template 우회), (2) 모든 호출이 `claim_provenance=None` → 항상 floor verdict, (3) `_PATTERNS`에 `strong candidate`·`paper-grade` 없음 → **CAR overclaim이 clean 통과**. `recommendation()`(`calibration.py:29-35`)은 geometry를 아예 안 봄. |
| **MechanismSpec ↔ Config** | first-class field 아님, ctx 주입 안 됨 | **partially_correct** | `Config.mechanism: Optional[MechanismSpec]` **이미 존재**(`config.py:627`), config-load 시 template hard-gate. s06가 ctx에 `mechanism_spec`/`geometry_terms` 발행(`s06:134-147`). **그러나 소비자 0개**(grep 확인) — dead-ended put. CAR config는 `mechanism:`를 설정조차 안 해 legacy `reactive_geometry` 경로로 돌았음. |
| **s04x reference ensemble** | ALL_STAGES에 없음/unwired + FDH 하드코딩 | **partially_correct** | `pipeline._effective_stages`가 `reference_ensemble.enabled`일 때 s04 뒤에 삽입(opt-in, default off) — unwired 아님. run/load parity OK. **맞는 부분**: `build_reference_ensemble_v0`가 `MechanismSpec` 미사용, `fdh_ref_v0_*` ID + 3.25 Å/165° 하드코딩, FDH seed가 default 경로. |
| **EvidenceCard canonical** | s11이 legacy scalar로 ranking | **expert_correct** | s11은 `final_score`(flat weighted sum, `nac_delta` weight 0.0)로 rank/library 선정. EvidenceCard는 non-fatal `try/except` side-car로 아무것도 안 먹임. `paper-grade:16` ∧ `rejected:16` 동시 표출은 `is_paper_grade`가 verdict를 안 봐서 나는 **실제 이중계상 버그**. |
| **s10 동적 NAC / Mg²⁺** | 4개 결함으로 occupancy=0 | **expert_correct (전부)** | 코드+run 데이터로 4개 모두 확인. Mg²⁺ 없음(반발→O가 3.94–4.70 Å 바닥), `transfer_is_h:false`에서 transfer==donor → 각도 구조적 NaN(`nac.py:139-147,220-251`), 제약이 carboxylate가 아닌 co-substrate COM(`openmm_engine.py:353-359`), production 25000 step=0.05 ns cap(`openmm_engine.py:27-54`). WT도 NaN → **유효 baseline 자체가 없음**. |
| **ML multi-head** | 여전히 단일 `ml_score` | **expert_correct** | `MultiHeadEvidencePrior`(real logreg, group-split, activity-head 금지)가 완성돼 있으나 src에서 **한 번도 인스턴스화 안 됨**. s08은 heuristic/XGBoost 단일 scalar. 요구 축(adenylation_geometry_prior 등) 전부 부재. |
| **mechanism templates** | 6개 중 2개만 real | **partially_correct** | **3개** real(hydride_transfer, nucleophilic_acyl_substitution, **glycosidic_bond_cleavage** — 전문가가 stub이라 한 건 틀림), 3개 stub. `adenylation_phosphoryl_transfer` 키 없음. 기존 `phosphoryl_transfer` stub은 note가 **Pγ(kinase)** — CAR의 Pα adenylation과 다른 화학. |

**결론**: v5는 "V3/V4 framework를 다시 만드는" 작업이 아니다. **이미 존재하는 배관을 실제로
소비시키고(consumer 연결), CAR 동적 NAC의 물리를 고쳐 유효 신호를 만드는** 완성 작업이다.
가장 큰 단일 leverage는 **adenylation mechanism template** — 이것이 framework의 빠진 template
이자 동시에 CAR 과학 결함의 수리처다(shared mechanism).

---

## 0.5 Round-2 정정 · 제품 계약(Product Contract) · 확정 결정 (2026-07-04)

### 0.5.1 git-ref 정정 — s08은 GitHub 기준 아직 blocker
`git rev-list --count origin/main..HEAD = 211`, `origin/feat/server-hardening..HEAD = 1`. s08 multi-lane
union은 **로컬 미푸시 커밋 `9ddf63f`에만** 있다. `origin/main`(`ad213ca`, "Initial … Phases 0-6")과
푸시된 `origin/feat/server-hardening` 둘 다 `s08_reranker.py`가 `candidates.sort(-ml_score)` 후
`top = candidates[:top_for_redocking]`인 **순수 top-cut**이다. 따라서 리뷰어가 GitHub에서 본 대로
**s08 boundary는 아직 해결되지 않았다.** 게다가 로컬 union조차 `ml_high+diversity+low_ml_control`만이라
요구 lane이 없다. `lane_allocator.py`는 `PlateBudget`(wt_replicates·known_active/inactive·negative_controls·
uncertainty_probes) + evidence lane(consensus_high·mechanism_geometry_high·cofactor_ligand_specific·
structural_evolutionary_high)을 **완비했으나 어떤 stage에도 연결되지 않았다**(grep `allocate(` in stages/ = empty).
→ **작업은 "top-cut을 union으로 전환 + 기존 allocator 연결", 신규 lane 로직 재작성 아님.**

### 0.5.2 제품 계약(Product Contract) 재프레이밍
V5는 "CAR를 잘 맞히는 파이프라인"이 아니라 **"어떤 효소에도 적용되는 claim-safe, mechanism-configurable
enzyme-variant triage platform"**을 목표로 하고, **CAR를 첫 고난도 acceptance test**로 쓴다. 단, **제품
규율(discipline)과 제품 포장(packaging)을 분리**한다.
- **지금 채택(규율)**: MechanismSpec가 모든 downstream stage의 기준 · EvidenceCard가 canonical output ·
  ClaimGuard가 platform invariant · ML은 evidence prior(후보 자르는 칼 아님) · reference state는 ensemble ·
  reproducibility bundle(config/artifact/tool/charge/protonation hash). 이 규율은 연구도구·논문·Cohort
  제출물로서도 EvoLiEZ를 더 낫게 만들므로 **상용화 여부와 무관하게** 채택.
- **보류(포장)**: Docker/API/CLI 안정화, template marketplace, 고객 UI, license audit — 구체적 이유
  (고객·grant·spinout)가 생길 때 명시적 결정으로. 특히 **license audit은 실제 gate**(guided_alphafold
  CC-BY-NC, ThermoMPNN/LigandMPNN/gnina/DiffDock 스택). 논문/Cohort(결과 2026-07-31) 트랙과 시간 경쟁하므로
  default로 흘러들지 말 것.

### 0.5.3 확정 결정 (사용자 지시 반영)
1. **`Config.mechanism` 이름 유지**(rename 금지 — config_sha1 purge trap). legacy `advanced.mechanism`(bool)과
   문서로 구분.
2. **`ReactionState`에 `cofactor_state: Optional[str]` 필드 추가**(ATP 등 non-redox cofactor 표현;
   `cofactor_redox_state`와 의미 겹치지 않게 문서화). 사용자 제안 config의 `cofactor_state: ATP`는 이 필드
   추가 전엔 로드 실패(`extra='forbid'`)이므로 **스키마를 먼저 수정**.
3. **`ligand_roles`는 `MechanismSpec`에 중복 정의하지 않음.** source of truth = `input.ligand.role` +
   `input.extra_ligands[*].role`. MechanismSpec는 reaction envelope(reaction class · reaction_state ·
   catalytic_residues · geometry_terms · geometry_calibration)로만 유지.
4. **mechanism 미선언 = legacy mode(hard-fail 금지)**. report에 legacy banner 강제("MechanismSpec not
   declared: running legacy geometry path. Reaction-geometry claims capped at uncalibrated/hypothesis-grade").
   strict production mode는 CAR V5 config·신규 template 안정화 후 별도 flag(`strict_mechanism_required`)로.
   `backend=real`만으로 hard-fail하지 않음. (검증: 현재 **어떤 config도** `mechanism:` block을 안 씀 →
   필수화 시 FDH 논문 config 전부 파괴.)
5. **Geometry source of truth = `mechanism.geometry_terms`.** `validation.md.reactive_geometry`는 legacy
   compat layer. s10 우선순위: `mechanism.geometry_terms` → (없으면) legacy `ReactiveGeometryConfig`를
   transient MechanismSpec로 lift → (없으면) no reaction-geometry evidence. report/provenance에
   `"geometry_source": "mechanism_spec | legacy_reactive_geometry | none"` 기록. **이중 정의 금지.**
6. **EvidenceCard 8축(v4)으로 통일**: structural_viability · evolutionary_tolerance · ligand_role_competence ·
   substrate_positioning · reaction_geometry_accommodation · reference_state_accommodation · off_pathway_risk ·
   uncertainty. v3 `rule_based_v1` confidence 이식 필요(L).
7. **EvidenceCard ranking driver 전환은 flag 뒤 default off** (`ranking.use_evidence_card: false`), A/B report
   (legacy_final_score_rank vs evidence_card_rank, rank_shift, lead_retained, control_retained) 생성 후
   **CAR Mg²⁺ dynamic NAC rerun 이후에 flip**.
8. **Mg²⁺ 1개부터**, coordination 유지가 acceptance. **CAR construct: A-domain/PCP(1–720) primary,
   full-length는 context validation**, A-domain adenylation screen에서 NADPH 제외 가능. (부수 이점: NADPH를
   빼면 P-보유 ligand가 ATP만 → car_nac α-P/leaving-O SMARTS 충돌 제거. 단 현재 `car_srcar_3hp.yaml`은
   `reactive_geometry.enabled: false`라 primary로 쓰려면 켜고 mechanism block 부착.)

### 0.5.4 확정 config 스케치 (스키마 정합)
```yaml
# top-level
mechanism:
  reaction: { class: adenylation_phosphoryl_transfer }
  reaction_state:
    substrate_state: 3HP_carboxylate
    cofactor_state: ATP            # ← NEW ReactionState field (결정 2)
    metal_state: Mg2+_bridged
    conformational_state: A_domain_closed
    protonation_model: pH7_4_manual
  catalytic_residues: ["S268","T269","K273","K629"]
  template_options: { require_mg_bridge: true }   # Mg term은 template엔 optional, CAR config에서 required 승격
  geometry_terms: []               # 비우면 template default 사용 (O→P distance + inline angle)
  # ligand roles는 여기가 아니라 input.ligand.role / input.extra_ligands[*].role
# strict enforcement는 config field가 아니라 ENV: EVOLIEZ_STRICT_MECHANISM=1 (purge-safe; 신규
# top-level Config field는 config_sha1 변경 → 기존 run purge. 헤더 노트 ⚠️ 참조)
```

### 0.5.5 Phase V5-C — Generality Proof (신규, 채택)
CAR 전용툴이 아님을 증명. **template marketplace는 구현하지 않음**(내부 asset로만).
1. FDH hydride-transfer **regression**(golden-file 불변).
2. TEM-1 nucleophilic-acyl-substitution **config-only smoke**.
3. glycosidase **config-only smoke**.
4. **template별 ClaimGuard policy**(mechanism마다 claim ceiling·금지 표현).
5. **mechanism별 assay schema**(wet-lab readout 정의).
완료 기준: **code edit 없이 config만으로** 3 mechanism run; 각 mechanism의 required reaction_state 누락 시
config-load fail; 모든 output이 동일 EvidenceCard schema.

---

## 1. 두 개의 병렬 트랙

### Track A — CAR-science (물리 유효성)
동적 NAC를 "생물학적 음성"이 아니라 **유효한 near-attack 가설 검증**으로 끌어올린다. 하나의
일관된 물리 결함(anion 반발 + degenerate angle + COM 제약 + 짧은 MD)의 네 얼굴을 고친다.
`car/car_nac.py`의 **정적 pass는 이미 올바른 물리를 인코딩**하므로 oracle로 재사용한다.

### Track B — framework claim-safety & completion (전역 무결성)
이미 만들어진 ClaimGuard·MechanismSpec·EvidenceCard·EvidencePrior·multi-lane 기계를 **실제
E2E 경로에 연결**한다. 핵심은 "구조적으로 overclaim 불가능"과 "mechanism-configurable가 런타임에
실제 작동"을 참으로 만드는 것.

두 트랙의 **수렴점**이 adenylation template(V5-3): Track A의 각도 물리를 담는 canonical home이자
Track B의 빠진 generic template.

---

## 2. Workstreams

각 workstream은 실제 file:line에 근거한 concrete delta + 로컬 테스트 가능한 acceptance +
dependency를 갖는다. **effort: S/M/L**, **priority: P0/P1/P2**.

### V5-1 · CAR 동적 NAC 물리 수리 (angle math + reactive-O 제약 + step-cap) — **P0 · Track A**
*Mg²⁺를 제외한 순수 코드 결함 3/4. 로컬 검증 가능. 최고 leverage.*

- **각도 진짜로 계산**: `transfer_is_h:false`일 때 degenerate donor–transfer–acceptor(=donor)
  대신 **O_nuc–Pα–O_leaving** 3점 각을 계산. `ReactiveSpec`/`nac_idx`를 3→4 atom으로 확장,
  leaving-O를 `leaving_smarts`(default ATP bridging β-O)로 해석하되 **공격 벡터에 가장 anti한
  O**를 선택(`car_nac.py:109-129`의 `oL = min(obr, key=λo: dot(o-pA, oatt-pA))` 그대로).
  H-transfer(FDH) 경로는 byte-identical 유지. — `md/nac.py:139-251`, `config.py`(ReactiveGeometryConfig).
- **reactive-O 제약**: `CustomCentroidBondForce`의 whole-cosub COM group을 `nac_map['donor_heavy']`
  단일 atom group으로 교체(`r0=3.6 Å`가 반응 O를 Pα에 실제로 고정). **distance-only 유지 —
  angle 제약 금지**(NAC 조작 방지). — `openmm_engine.py:339-361` + call site.
- **step-cap 분리**: NAC 관련 production을 25000 step에 가두지 말 것. focused NAC lane 전용
  `nac_production_ns`(또는 protocol_level 3) 도입, top-N만 2 ns. — `openmm_engine.py:41-54`,
  `s10_md.py:542-544`, `car_srcar_3hp_full.yaml`.
- **subframe 3→4 atom**: leaving-O global index를 매 frame 수집. — `openmm_engine.py:~1324-1368`.
- **Acceptance (로컬)**: 합성 4-atom in-line fixture → 각도≈180 + occupancy>0; 옛 3-atom 경로는
  NaN 회귀 lock; restraint force가 단일 atom group == donor_heavy이고 `angle(` 무존재(grep);
  new NAC tier `_production_nsteps`≥1e6(2 ns) + `actual_ns==2.0`; FDH hydride 테스트 불변.
- **depends_on**: 없음.

### V5-2 · Mg²⁺ co-modeling (WT + 모든 mutant 동일) — **P0 · Track A**
*V5-1 각도를 고쳐도 Mg²⁺ 없이는 anion 반발로 O가 ~4 Å 바닥 → occupancy 0. 물리적 전제조건.*

- Mg²⁺를 `role: metal` ligand로 추가(pose_gate에 metal role 이미 존재). +2 point charge(bonded
  term 없음), `_apply_fixed_charges` + SystemGenerator molecule set에 등록. ATP α/β-phosphate와
  3-HP carboxylate를 bridging하도록 1–2개 배치(coordination seed; 선택적 **distance-only** weak
  Mg–phosphate 제약). — `openmm_engine.py:~908-960`, `config.py`(LigandSpec), `car_srcar_3hp_full.yaml`.
- **build parity**: `build_anchored_mutant_complex`가 reference를 deepcopy하므로 Mg set 상속 —
  단, anchored PDB round-trip + CONECT remap(H 제거, het→LIG 개명)에서 Mg 생존 검증. **WT와 모든
  mutant가 동일 Mg 배치**여야 ΔNAC 유효. — `md/anchored_build.py`, `md/cosubstrate_placement.py`.
- **Acceptance**: 소형 합성 complex 최소화 blow-up 없음(NaN energy 없음); WT reference build와
  deepcopy mutant build의 Mg atom 수·index 동일; (server-gated) Mg+각도 수리 후 WT가 일부 frame에서
  3.6 Å 아래로 dip 가능한 유효 baseline 생성.
- **depends_on**: V5-1.

### V5-3 · `adenylation_phosphoryl_transfer` mechanism template (framework ∧ CAR) — **P0 · both**
*shared-mechanism workstream. 기존 `phosphoryl_transfer`(Pγ) stub을 덮지 말 것 — 다른 화학.*

- **새 template 키** `adenylation_phosphoryl_transfer` 추가. `required_reaction_state=
  [metal_state, conformational_state, substrate_state]`. default_geometry_terms:
  (1) distance `nuc_O_to_alphaP` `[OX1-]`→`[PX4]([OX2][CX4])` max 3.6; (2) angle
  `inline_attack_Onuc_Pa_Oleaving`(vertex α-P, c=bridging β-O) 150–180 **read-only**;
  (3) **선택적** distance `mg2p_coordination`(metal_state gated, **required 아님** — Mg 없으면
  hard-abort 방지); (4) distance `K629_a3loop_contact_retention`. — `mechanism/templates/__init__.py`.
- **CAR config에 `mechanism:` block** 추가(`reaction.class: adenylation_phosphoryl_transfer`,
  `reaction_state{substrate_state:3HP_carboxylate, conformational_state:A_domain_closed,
  metal_state:Mg2+_bridged}`) → `Config.mechanism` 설정 → s06가 `mechanism_spec`/`geometry_terms`
  발행. **주의**: `ReactionState`에 `cofactor_state` field 없음 — `cofactor_redox_state` 재사용 또는
  field 추가(`extra='forbid'`가 미지 키 거부). — `car_srcar_3hp_full.yaml`, `mechanism/spec.py`.
- **car_nac.py 로직 이식**: `alpha_P()`/`leaving_dir()`/`carboxylate()`/`wrong_pose()` selector를
  template runtime의 frame-0 placement validity gate로. **heavy-atom-count heuristic → SMARTS/residue
  해석으로 전환**(full 3-ligand complex에선 NADPH도 P 보유). — `md/nac.py`, `md/cosubstrate_placement.py`.
- **Acceptance**: `get_template('adenylation_phosphoryl_transfer')`가 O→P term 반환, 기존
  `phosphoryl_transfer` stub 불변; metal_state unset+adenylation class는 config-load hard-abort;
  ported selector가 SMARTS 기반으로 flipped-3HP pose(hydroxyl이 carboxylate보다 α-P에 가까움) 거부;
  CAR config load 후 s06가 ctx에 두 키 발행.
- **depends_on**: V5-1.

### V5-4 · MechanismSpec 소비자 연결 (dead put → 실작동) — **P1 · Track B**
*`Config.mechanism`·s06 발행은 있으나 소비자 0개. V5-3 template이 런타임 효과를 갖게 하는 일반화.*

- s09/s10/`md.nac`가 `ctx.get('geometry_terms')`/`ctx.get('mechanism_spec')`를 읽어 MechanismSpec
  geometry term + reaction_state로 scoring(legacy 경로 앞단에 gating). **`config.mechanism is None`이면
  legacy `features.mechanism`/`ReactiveGeometryConfig` 경로 byte-identical fallback** — 재고 FDH config
  불변. — `s09_nonmd_validation.py`, `s10_md.py`, `md/nac.py`.
- **PARITY GUARD**: s06는 현재 `load()` override 없어 --resume 시 항상 re-run(base.load()→False) —
  오늘 parity 버그 없음. **s06.load()를 추가한다면 run()의 모든 ctx.put 키를 복원해야 함**
  (mechanism_spec/geometry_terms + interaction_graph/contacts/residue_table/designable_positions/…).
  안 하면 --resume 시 s09/s10이 조용히 legacy로 회귀. run()/load() 키-parity 테스트 필수.
- **Acceptance**: mechanism 설정 시 s09/s10가 ctx geometry_terms로 scoring, `mechanism=None`이면
  legacy와 golden-file byte-identical; `grep ctx.get('geometry_terms')` non-empty(전엔 empty);
  s06.load() 미추가(또는 추가 시 키-parity 테스트 통과).
- **depends_on**: V5-3.

### V5-5 · ClaimGuard: geometry-gate + CAR pattern + real provenance — **P0 · Track B**
*CAR overclaim("strong candidate" ↔ "geometry 0.00")을 구조적으로 불가능하게. 대부분 소형·로컬.*

- **recommendation() geometry gate**: `ts_geometry_score`(및/또는 NAC occupancy)를 signature에
  추가, geometry가 0(dead)/None(unknown)이면 `strong candidate` 금지 — 진짜 양의 geometry만 strong
  유지. s11 call site에서 `c.scores.get('ts_geometry_score')` 전달. **소스에서** CAR 케이스 차단.
  — `calibration.py:29-35`, `s11_final_ranking.py:84`.
- **CAR 패턴 추가**: `_PATTERNS`에 UNQUALIFIED_STRENGTH — bare `strong/top/promising candidate`는
  structural/binding 수식 없으면 금지, `paper-grade`는 claim 단어로 금지(→ `anchored-evaluated`).
  HTML·markdown 경로가 `_PATTERNS` 공유하므로 1회 수정으로 양쪽 커버. — `claim_guard.py:36-78`.
- **paper-grade/rejected 이중계상 수리**: `is_paper_grade`가 gate_stack verdict ≠ REJECTED 요구
  하도록. wt_anchored+reference_like인데 MD stability 실패한 후보가 두 집합에 동시 계상되던 버그 제거.
  — `evidence.py:34-44`.
- **real provenance 주입**: `io/_provenance_card.py` 신설 — **on-disk run state**(reports/provenance/*.json,
  _state.json; in-memory ctx 아님 → --resume regen 동일)에서 `ClaimProvenance` 재구성, 6개 `page()`
  site + `report.py` markdown에 전달(`evaluate(None)`→`evaluate(built)`). **fail-safe**: 누락/malformed는
  over-restrict만 가능; wetlab_replicated=True 없이 activity/kcat 절대 unlock 금지, unlock은 **검증된
  server 구조**로만(v4 synthetic fixture 금지). — `io/{md,paper,paper_report_v2,rerank,validation}_report.py`, `report.py`.
- **copy 수정**: `strong candidate`→`strong structural/binding candidate`, `paper-grade`→`anchored-evaluated`.
- **Acceptance**: geometry=0 fixture에서 recommendation()이 strong 아님(None도 아님); `lint_text('strong
  candidate')` FAIL / `'strong structural candidate'` PASS / `'paper-grade'` FAIL; rejected+wt_anchored+
  reference_like가 `is_paper_grade==False`; CAR report 재생성이 banner-clean + bare 'strong candidate' 무존재
  (CI에서 `EVOLIEZ_STRICT_CLAIMS=1`로 hard-fail).
- **depends_on**: 없음.

### V5-6 · ClaimGuard 커버리지: 8개 자체-template report — **P2 · Track B**
- 공유 choke-point `kit.assert_html_clean(html, provenance, title)` 추가, 8개 `write_*_report()`
  말미에서 호출(전면 port보다 저렴). `_strip_tags`가 script/style 제거하므로 body-text lint. —
  `_report_kit.py` + `{complex,docking,homolog,input,interaction_model,msa,provenance,s07}_report.py`.
- **Acceptance**: 8개 모듈 `claim_guard` grep non-zero; bare 'strong candidate' 주입 시 guard 발동;
  기존 report-gate 테스트 통과.
- **depends_on**: V5-5.

### V5-7 · s08 top-cut → 완전 lane union + `lane_allocator` 연결 — **P0 (재분류) · Track B**
*⚠️ round-2 정정: GitHub 기준 s08은 아직 순수 top-cut(§0.5.1). 로컬 minimal union도 요구 lane 없음.
`lane_allocator.py`는 요구 lane을 완비했으나 unwired. → **top-cut을 union으로 전환 + 기존 allocator 연결**.*

**필수 lane set** (사용자 지시): `ml_high` · `mechanism_geometry_high` · `ligand_role_competence_high` ·
`structural_evolutionary_high` · `diversity` · `uncertainty_probe` · `low_ml_control` · `deconvolution` ·
`known_controls`. CAR 추가: `static_adenylation_near_attack_high` · P438/G430/G407 loop deconvolution ·
`scalar_top_control`.
- s08b fold/redock queue를 `lane_allocator.allocate()`(controls 먼저 예약 → evidence lane priority →
  diversity, PlateBudget으로 overflow 없음)로 구성. **union-not-shrink 유지**(ml_high ≥ 기존 top_for_redocking).
- low-ML control lane이 비면 run **fail**(사용자 지시). known active 없어도 abort 금지(uncalibrated flag).
- deconvolution은 candidate-EXPANSION(파생 sub-mutant를 ctx candidates 주입 + --resume persist).
- lane 태그 persist(s11 `lane_by_id`가 `details['selection_lane']` 읽음).
- **재분류 근거**: GitHub 코드가 top-cut이므로 이건 "완성"이 아니라 실제 blocker → P1→**P0**.
*(참고: 아래 원안은 "union은 이미 있으니 lane만 ADD"였으나 §0.5.1 정정으로 무효화됨.)*

- **catalytic_static_high lane**: s08-**계산 가능한** static reacting-atom geometry score로 keying
  (동적 NAC는 s09/s10 product라 stability/geometry처럼 inert). s08/s08b LaneConfig에서 s09-only면
  0으로. **모든 후보에 score 부재 시 fail-loud/log**(silent 0-기여 함정). CAR은 Mg²⁺ 없인 동적
  NAC=0이므로 **static NAC 신호를 써야 판별력**. — `multi_lane.py:74-101`, `config.py`, s08b/s09.
- **known/controls**: 이미 만들어진 `lane_allocator.allocate()`(known_active/inactive/negative 예약 +
  96-well PlateBudget)를 s08b fold/s09 MD에 연결(또는 경량 known-controls lane). per-lane count와
  double-count 조율; known active 없어도 abort 금지. — `lane_allocator.py:100-165` + call sites.
- **deconvolution lane**(candidate-EXPANSION): multi-point lead마다 single/pairwise sub-mutant 파생,
  `details['selection_lane']='deconvolution'` + `deconvolved_from=<lead_id>` 태그. **파생 sub-mutant는
  ctx candidates에 먼저 주입 + --resume 결정성 위해 persist**해야 s08b fold 가능. — `multi_lane.py`,
  `s07_mutation_gen.py`, `config.py`.
- **lane 태그 persist**: reranked/validated_candidates.json에 all_lanes 멤버십 기록(s11 `lane_by_id`가
  `details['selection_lane']` 읽음 → persist 안 하면 resume 시 '?'). — `s08_reranker.py:283`, s09, `s11:138`.
- **Acceptance**: catalytic-static lane이 ml_high가 버릴 후보를 추가하고 union이 baseline 아래로 안 줄음
  (union-not-shrink assert); score 전부 부재 시 log/raise; allocate()가 known-active/WT-replicate 예약
  + double-count 없음 + known active 없어도 무-abort; deconvolution sub-mutant가 ctx 주입 + persist +
  모의 --resume 생존; s11 lane_by_id가 새 lane 표시.
- **depends_on**: 없음.

### V5-8 · EvidenceCard를 s11 canonical ranking driver로; final_score는 provenance 컬럼으로 강등 — **P1 · Track B**
*가장 큰 ranking-semantics 변경. P0 물리/claim 이후. 기존 run A/B 필요.*

- 모든 ranked 후보에 **main 경로**로 per-candidate EvidenceCard 생성(swallow side-car 아님),
  `c.details['evidence_card']` 부착, card-build 실패는 **fail-loud**(silent warn 금지). **한 impl로
  통일** — v4 `EvidenceCardV4`(8축, per-axis score/confidence/evidence/provenance + reference_state_
  accommodation)로 통일하고 v3의 `rule_based_v1` confidence 모델 이식(또는 v3를 s11 card로 명시).
  — `s11_final_ranking.py`, `evidence_card{,_v4}.py`, `triage_report.py`.
- evidence class 내부 rank(confirmed>improved>no_gain>alternative_pose>rejected) + axis-aware tie-break;
  focused library를 `experimental/library_plan.propose_round2_library`(이미 `reject_final_score_only`)로
  라우팅. **final_score는 back-compat용 컬럼으로만 유지**. 주의: `nac_delta` weight 0.0이었으므로 axis
  ranking은 실제로 ΔNAC를 사용 → 결과 이동. CAR/FDH lead(S340G/G430R;S433F;G407K) 여전히 surface하는지
  A/B. — `s11_final_ranking.py`, `active_learning.py`, `library_plan.py`, `candidate_accommodation.py`.
- **PARITY**: card가 ranking driver가 되면 s11.load()가 card 유도 score/rank를 run()과 동일 복원.
  reaction_geometry 축은 **ΔNAC-vs-WT(MD 유도)**, Boltz-pose `catalytic_geometry_penalty`(오염, confidence-only)
  아님. confidence는 score와 독립 유지. 텍스트 ClaimGuard-clean.
- **Acceptance**: 모든 후보에 evidence_card; 강제 실패는 raise/flag; 저장된 fdh/car snapshot A/B에서 lead가
  top set에 잔존; focused library가 propose_round2_library 경유(final_score-only 입력은 raise);
  s11.load() rank/card score value-parity; reaction_geometry 소스가 ΔNAC.
- **depends_on**: V5-1.

### V5-9 · 축 분리 final table + config-driven reaction 라벨 — **P2 · Track B**
- evidence_cards.json에서 축별 score+confidence table 렌더(structural viability | ligand/ATP/NADPH
  retention | 3-HP carboxylate orientation | O→P near-attack dist+angle | Mg bridge | uncertainty).
  reaction-specific 라벨→generic 축 매핑은 `terminology.py`(display 개명 home), FDH/3HP 문자열
  하드코딩 금지. `v4_candidate_card.render_candidate_cards`를 table로 확장. final_candidates.csv에 축 컬럼.
- **Acceptance**: 축 table이 per-axis score+confidence 렌더 + ClaimGuard-clean; 라벨이 config/target
  map 유래(report 코드에 'carboxylate'/'Mg bridge' 리터럴 grep empty); csv에 축 컬럼 + final_score 강등 잔존.
- **depends_on**: V5-8.

### V5-10 · mechanism-plausibility EvidencePrior head를 s08 ADDITIVE 신호+lane로 — **P2 · Track B**
*server-retrain-gated, 최저 우선순위. ml_score에 절대 합산 금지(binding-only 의미 이중계상 방지).*

- HEAD_SPECS에 **cheap·MD-free geometry/consensus surrogate** head 추가(adenylation_geometry_prior,
  atp_alphaP_alignment, mg_bridge_compatibility, flip_risk=reference_like-vs-displaced pose_gate,
  off_pathway_binding). `learnability.LABELS`+CHEAP_PREFIXES 등록(assert_no_leakage 통과),
  FORBIDDEN_HEAD_TOKENS(kcat/km/activity/…) 유지. — `evidence_prior.py`, `learnability.py`.
- 훈련된 artifact 있을 때만 s08에서 per-head prior 계산 → `mech_plausibility_score` + per-head를
  cand.scores에 별도 저장(**ml_score에 합산 금지**). config flag + artifact 없으면 graceful no-op.
  mech_plausibility lane 추가. — `s08_reranker.py`, `multi_lane.py`, `s08b`.
- prior_eval로 head별 top-k enrichment/FN-rate/lane_rescue → binding-only baseline 초과 enrich할 때만
  lane 부여. survivorship(negative 부족) 때문에 라벨은 **전체 후보 집합**에서 생성.
- **Acceptance**: 새 feature 모두 assert_no_leakage 통과; forbidden token head는 raise; artifact 없으면
  s08 byte-identical no-op; mech_plausibility_score 별도 저장(ml_score/ranking.score 이중 미포함);
  s08.run()이 키 추가 시 load()가 복원.
- **depends_on**: V5-3, V5-7.

### V5-11 · s04x v0 builder generic화 + FDH seed 재배치 — **P2 · Track B**
- `build_reference_ensemble_v0`가 `MechanismSpec` 수용 → geometry_terms에서 per-term center 유도
  ((distance_min+max)/2, angle_min/max), 하드코딩 3.25/165 + 고정 8-row offset 제거. distribution을
  GeometryTermSpec.label로 keying(literal 키 아님) — 모든 consumer lockstep 업데이트. — `guided/ensemble_builder.py`.
- conformer_id prefix generic화(`{target_id}_ref_v0_{i}`), FDH ctx 키(nadp_anchor/formate_reactive)를
  generic `reference_term_distributions`로. `CTX_KEYS` 단일 source로 parity 자동 유지. artifact
  schema_version bump. `_data_provenance=seed_fixture_v0`/`computed_from_this_run=False` 스탬프 **유지**(fixture).
  ≥8 conformer 또는 insufficiency_reason. — `s04x_reference_ensemble.py`.
- FDH seed를 default 경로 밖으로: `DEFAULT_SEED_MANIFEST=None`(명시적 config/env 강제),
  mut_00479/Q382R invariant을 per-target YAML `invariants` block으로. — `seed_manifest.py`, `examples/fdh/`.
- **Acceptance**: 비-hydride MechanismSpec가 그 mechanism term 중심 conformer 생성(3.25/165 아님) + ≥8 +
  provenance; seed_fixture_v0 스탬프 유지; `test_b6_generic_manifest_accepts_non_fdh` 통과; stale
  구-format artifact가 schema_version check로 re-run.
- **depends_on**: V5-3, V5-4.

---

## 3. CAR 재실행 프로토콜 (Mg²⁺ + 수정 O→P geometry)

**후보 집합 (6–8 focused + deconvolution)**:
- **WT SrCAR** — 필수 baseline. 동일 Mg²⁺ 배치 + 수정 각도 필수(현재 WT는 angle=nan/invalid → baseline 없음).
- **G430R;S433F;G407K** — 완료 run의 multi-point catalytic lead.
- **P438N** — clean single(전문가 지목).
- **G430R / S433F / G407K** — lead deconvolution single.
- **G430R;S433F**(및 1개 pairwise) — 효과 귀속.
- **negative/off-pathway control 1개** — screen이 LOW score를 낼 수 있음을 확인.

**단계**:
0. **PURGE-safe**: `mechanism:` block + Mg²⁺ + nac_production tier는 config_sha1 변경 → 라이브
   `srcar_3hp_full` run에 `ctx.setup()` 금지. 격리 dir 사용 또는 `_state.json` fingerprint를
   `scripts/check_run_fingerprint.py --patch`로 선-패치(EVOLIEZ_ALLOW_PURGE). focused smoke는 anchored
   PDB에서 WT complex 직접 build.
1. V5-1 착지 + **로컬 검증**: 4-atom in-line fixture → 유한 각도≈180 + occupancy>0; FDH 3-atom 불변.
2. V5-2(Mg²⁺) + V5-3(adenylation template + CAR `mechanism:` block, mg2p_coordination OPTIONAL) 착지.
   deepcopy mutant build가 동일 Mg 배치 상속 검증(parity 테스트).
3. reactive-geometry/NAC lane만 protocol_level 3(또는 nac_production_ns=2.0), 나머지는 0.05 ns screen
   유지. **core ≤16–24, concurrency gate**(shared-box watchdog).
4. WT + focused(6–8) replica로 s10 실행, 4-atom subframe(O_nuc, Pα, O_leaving, acceptor) 수집.
   md_candidates.json에 유한 angle_mean + 일부 frame에서 3.6 Å 아래 dip 가능한 distance_min 확인.
5. **유효 WT baseline** 대비 ΔNAC = mutant_occ − WT_occ 계산. occupancy는 diagnostic(distance는 제약
   편향, angle은 read-only). 동일 pose로 static `car_nac.py` oracle 교차검증.
6. focused set을 evidence class 내부(V5-8)에서 ΔNAC-vs-WT reaction_geometry 축으로 rank(legacy
   final_score 아님). ClaimGuard-clean 축 분리 report 재생성(V5-9), geometry-gated recommendation(V5-5).
7. wet-lab deconvolution library(lead + singles/pairwise + P438N + negative control)를 known-control-aware
   `lane_allocator`(V5-7)로 16–24-well plate 발행.

**성공 기준 (⚠️ "lead가 나오면 성공"이 아니라 "system이 올바른 evidence를 내면 성공")**:
System은 다음이면 **PASS** — discrimination은 hard 조건이 **아님**(방법론 수정으로 보장 불가한 생물학적 outcome):
- angle finite(no NaN) — 진짜 O_nuc–Pα–O_leaving 3점 각(read-only, angle 제약 grep-empty).
- Mg²⁺가 ATP-phosphate/3HP-carboxylate 근처에 frame-wise 유지(빠지면 "low activity"가 아니라 **invalid
  geometry setup**으로 기록).
- WT baseline finite(invalid_cosubstrate_diffused 아님) → 실 baseline 존재.
- 모든 focused 후보의 geometry term이 계산됨.
- **discrimination이 안 나와도**: 원인 정량 진단(예: 2 ns implicit-solvent 미샘플링) + escalation route
  (static+longer-MD / QM-MM) 기록 + ClaimGuard가 overclaim 차단 → **그래도 PASS**.
운영 지표(달성 시 보고, 미달 시 진단):
- Mg²⁺ 하에 일부 후보의 일부 frame에서 reactive-O–Pα < 3.6 Å(~4 Å 바닥 돌파).
- ΔNAC-vs-WT가 focused 후보 전부 유한; lead + deconvolution single의 ΔNAC로 귀속.
- 재생성 report가 banner-clean + bare 'strong candidate'/'paper-grade' 무존재; geometry 0/None 후보에
  'strong' 미부여(✅ V5-5 착지분으로 이미 강제됨).
- focused NAC set이 실제 2 ns(actual_ns==2.0).

---

## 4. Sequencing

**두 P0 트랙을 1일차부터 병렬로.**

- **Track A (물리)**: V5-1(순수 코드·로컬) → V5-2(Mg²⁺·server-gated) ∥ V5-3(adenylation template) →
  V5-4(소비자 연결). V5-1이 먼저여야 함(oracle-backed 로컬 코어; V5-3 template geometry는 각도 수리 전엔 무의미).
- **Track B (claim-safety)**: **V5-5를 즉시 시작**(가장 저렴·로컬·고무결성; 물리 진척과 무관하게 CAR
  overclaim을 구조적으로 차단하는 최속 risk-reducer). V5-6은 V5-5의 공유 `_PATTERNS`/choke-point 뒤.
  V5-7(lane)은 양 트랙과 독립 병렬(union 기계 존재, lane만 ADD).
- **수렴**: V5-8(EvidenceCard ranking)은 reaction_geometry 축이 수정 ΔNAC를 소비하므로 V5-1 의존;
  V5-9(축 table)는 V5-8 뒤; V5-10(EvidencePrior)은 V5-3+V5-7 의존·server-retrain-gated라 배선 작업 중 최후;
  V5-11(s04x generic화)은 V5-3+V5-4 의존·opt-in이라 최저 우선순위.

**V5-5 front-loading 이유**: 가장 싸고·로컬이고·고무결성이며 물리 진척과 무관하게 모든 하류 report를 de-risk.

---

## 5. Merge 전략 — 지금 main vs. v5 tag gate

**지금 main으로 (server-hardening core, 저위험·로컬 검증·ranking-semantics 무변경)**:
- **V5-1** (NAC 각도 + reactive-O 제약 + step-cap tier — `transfer_is_h:false` 경로 뒤라 FDH 불변)
- **V5-5** (recommendation geometry gate + ClaimGuard pattern + is_paper_grade 수리 + copy — 순수 무결성)
- **V5-6** (report 커버리지 choke-point)
→ 모두 additive/guard-only, 재고 FDH config byte-identical, 각자 로컬 회귀 테스트.

**v5 / "v3-complete" tag를 gate (전부 착지 + 실 run server 검증 필수)**:
- **V5-2** Mg²⁺(실 coordination geometry + server 최소화), **V5-3** adenylation template + CAR mechanism
  block(CAR config_sha1 변경 → NAC 재실행으로 non-degenerate 신호 증명), **V5-4** MechanismSpec 소비자(실
  ranking-input 변경 → FDH golden-file parity + CAR run), **V5-8** EvidenceCard canonical(top_candidate/DB
  rank/plate 이동 → 저장 fdh+car snapshot A/B로 lead 잔존 확인), **완료된 CAR 재실행**(유효 WT baseline + 유한 ΔNAC).

**tag 이후 opt-in follow-up (tag 미차단)**: V5-7(lane, deconvolution-expansion은 ctx 주입+resume persist
증명 필요), V5-9(report-only), V5-10(retrain-gated·default no-op), V5-11(opt-in stage·현재 honest stamp).

**tag CI hard gate**: `EVOLIEZ_STRICT_CLAIMS=1`로 ClaimGuard warn+banner→hard fail; v5에서 ctx.put이 추가된
모든 stage에 run()/load() 키-parity 테스트.

> **⚠️ merge claim 규율**: 이 브랜치를 "V3/V4 complete"로 합치지 말 것. "server/real-backend hardening +
> V3 schema/wiring; mechanism-configurable triage in completion"으로. ClaimGuard가 우리 문서에도 적용됨.

---

## 6. 열린 결정 (권장안 포함 — 미회신 시 권장안 진행)

1. **`Config.mechanism` 이름**: 현행 유지(권장). rename은 car/fdh YAML + config_sha1(purge trap) 파괴.
   legacy `advanced.mechanism`(bool)과의 구분은 문서로.
2. **ReactionState `cofactor_state`**: adenylation ATP_Mg는 `cofactor_redox_state` 재사용(권장, 스키마 무변경)
   vs. 새 field 추가. `extra='forbid'`라 미지 키 불가.
3. **Mg²⁺ parameterization**: +2 bare point charge(권장 시작) vs. weak Mg–phosphate distance 제약(distance-only).
   개수(1 vs 2)와 coordinating seed(A3 loop S268/T269/K273 + α/β-phosphate) — **CAR 생화학 판단은 사용자 몫**.
4. **V5-8 flip 방식**: axis-ranking을 config flag(default off)로 tag 넣고 CAR/FDH lead 잔존 확인 후 flip(권장)
   vs. 즉시 default. ΔNAC(weight 0.0였음)를 실제 사용하므로 top candidate 이동.
5. **v3(6축) vs v4(8축) EvidenceCard 통일**: v4로 통일 + v3 confidence 모델 이식(권장 — reference_state_
   accommodation 축이 전문가 요구와 일치) vs. v3 유지 + v4 experimental 표기. confidence 이식량 결정.
6. **CAR deconvolution plate**: P438N가 유일 clean single인지, negative/off-pathway control 후보 지정 —
   **wet-lab 결정은 사용자 몫**.
7. **FDH seed 재배치(V5-11)**: tag 전 `examples/fdh/`로 이동(암묵 default caller 파괴) vs. post-tag 연기
   (권장 — s04x는 opt-in·honest stamp).

---

## 7. Guardrails (v5 구현자 필독)

- **resume-load-parity**: stage가 run()에서 하는 모든 ctx.put을 load()가 복원해야 함. v5에서 ctx.put 추가
  시 run()/load() 키 diff 테스트. s06에 custom load()를 추가한다면 mechanism_spec/geometry_terms 포함 전 키 복원.
- **config_sha1 purge trap**: mechanism/Mg²⁺/weight/flag 추가는 config_sha1 변경 → 라이브 run에 `ctx.setup()`
  하면 rmtree. `_state.json` fingerprint 선-패치 또는 격리 dir(EVOLIEZ_ALLOW_PURGE guard).
- **angle-restraint 금지**: near-attack 각도는 **read-only occupancy 전용**. 어떤 각도 제약도 NAC를
  제조(manufacture)하므로 금지. reactive-O distance flat-bottom만 허용.
- **leakage guard**: 새 ML feature는 `learnability.CHEAP_FEATURES`에 있고 docking/MD 이전에 계산 가능해야
  `assert_no_leakage` 통과. mechanism head는 **plausibility prior**, activity predictor 아님(FORBIDDEN_HEAD_TOKENS).
- **local-first**: V5-1/V5-5/V5-6과 V5-2/V5-3/V5-4/V5-8의 로컬-검증 가능 부분은 `.venv-light` + 합성
  fixture + 저장 snapshot으로 **먼저** 검증. server 시간은 Mg²⁺ 최소화·실 NAC 재실행·golden-parity에만.
- **honest-stamp 유지**: s04x `seed_fixture_v0`/`computed_from_this_run=False`, MD `requested vs actual_ns`,
  NAC `distance는 제약 편향/angle read-only` — 모든 diagnostic 스탬프 보존.
