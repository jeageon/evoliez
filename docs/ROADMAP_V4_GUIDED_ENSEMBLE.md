# ROADMAP V4 — Active-State Reference Ensemble & Guided Triage

> **한 줄 요약**: 단일 예측 구조를 믿지 말고, 구조 예측 모델을 *prior*로 낮춘 뒤,
> 실험·문헌·기하 제약을 *likelihood*로 넣어 **active-state reference ensemble**을 만들고,
> 그 ensemble에 후보 돌연변이가 얼마나 잘 맞는지(accommodation)로 triage 한다.

- **상태**: DESIGN (미구현). v3 논문 마감과 **분리된 별도 트랙**.
- **근거 논문**: Maddipatla et al., *"Experiment-guided AlphaFold3 resolves measurement-consistent
  protein ensembles"*, Nature Biotechnology (2026).
  <https://www.nature.com/articles/s41587-026-03166-5>
- **참조 코드**: `sai-advaith/guided_alphafold` (GitHub) — **vendor 금지, clean-room 참고만**.
- **선행 문서**: `ROADMAP_V3_MECHANISM_TRIAGE.md`, `ML_V2_POSE_LABEL_INVESTIGATION.md`.
- 관련 메모리: anchored-validation-redesign, ml-v2-redesign, s10-nac-catalytic,
  evoliez-v3-implementation.

---

## 0. 왜 V4인가 — v3의 구조적 병목

v3까지 우리 파이프라인의 진실 원천은 **WT 단일 pose 하나**였다. 이 단일 reference가
NAC, pose gate, ML label, ClaimGuard geometry-evidence를 **전부** 파생시킨다.
따라서 reference가 틀리거나 sparse하면 하류 전체가 조용히 오염된다.

관측된 실제 증상 (메모리 기록):

- **single WT reference 오염**: per-mutant Boltz가 NADP를 WT 대비 8.3 Å 튀게 예측.
- **de novo pose 과민**: mutant Boltz pose는 대부분 `displaced`, anchored MD는 반대로 전부
  `reference-like`. 둘 중 하나를 truth로 삼을 근거가 없었다.
- **WT NAC sparse**: WT에서도 NAC가 낮으면 mutant의 ΔNAC를 binary pass/fail로 못 쓴다.
- **false negative 위험**: top-1 pose / top-1 candidate만 남기면 evidence가 다른 좋은 후보를
  일찍 버린다.
- **비물리적 붕괴**: reference/anchored 구조가 clash·bond strain을 안고 통과할 수 있다.
- **과대 주장 위험**: "functional-state proof"라는 표현이 posterior 가설을 population처럼
  들리게 한다.

논문은 정확히 이 6가지를 다룬다. V4는 논문의 **철학과 소프트웨어 패턴**을 이식한다
(효소 반응성 도구 자체를 이식하는 것이 아니다).

---

## 1. 논문에서 배우는 6가지 (문제 → 논문 전략 → 우리식 번역)

| # | 우리 문제 | 논문 전략 | 우리식 번역 |
|---|---|---|---|
| 1 | 단일 WT reference가 틀리면 NAC·gate·label 전부 오염 | AF3를 정답 생성기가 아니라 **sequence-conditioned prior**로 두고, 관측값을 likelihood로 넣어 measurement-consistent **ensemble** 생성 | `single WT reference` → **`ActiveStateReferenceEnsemble`** |
| 2 | de novo pose 과민 (displaced vs reference-like 충돌) | de novo prediction을 hard label로 쓰지 않고, diffusion 중간에 관측 likelihood gradient로 방향 조정 (prior+evidence 동시 작동) | `de novo displaced = inactive` ✗ → **`unguided disagreement = uncertainty signal`** |
| 3 | WT NAC조차 sparse → ΔNAC binary 불가 | 관측값을 단일 cutoff가 아니라 **ensemble-averaged observable**로 평가 (NOE는 ensemble 평균 거리) | `NAC hard cutoff` → **`reactive-geometry distribution / soft kernel / ensemble fit`** |
| 4 | ML·docking gate가 좋은 후보를 일찍 버림 | sampling 후 **ensemble pruning** — 여럿 만들고 관측값을 가장 잘 설명하는 작은 subset만 남김 | `top-1 pose / top-1 candidate` → **`multi-lane ensemble + pruning`** |
| 5 | 구조가 비물리적으로 깨짐 | guided sampling 후 반드시 **force-field relaxation + validity term** | guided ensemble도 반드시 `clash·bond·strain·OpenMM/Amber relax` 통과 |
| 6 | "이게 진짜 population인가?" | 생성 ensemble은 thermodynamic equilibrium이 아니라 **posterior structural hypothesis**라고 못박음 | `functional-state proof` ✗ → **`measurement/reference-consistent structural hypothesis`** |

---

## 2. guided_alphafold 코드 해부 — 이식할 6개 프리미티브

레포 조사 결과, 그들의 "experiment-guided diffusion"은 **모델-무관한 6개 부품**으로 분해된다.
guidance 전체가 사실상 **2개 파일**(`experiment_manager.py` + `get_x_t_from_x_0_hat`)에
담겨 있어 다른 diffusion 백엔드(Boltz)로의 이식이 현실적이다.

| ID | 프리미티브 | 위치 (guided_alphafold) | 핵심 |
|---|---|---|---|
| ① | **Guidance 샘플링 루프** | `experiment_manager.py:265-324` | 매 스텝: `x_0_hat=denoise(x)` → `loss.backward()` → `grad`를 다음 스텝에 주입 |
| ② | **Gradient 주입 수식** | `non_diffusion_model_manager.py:444-461` | `delta += step_size · normalize(grad)`; `denoiser_time_index` 이후에만 켜서 "전역 fold 먼저 → 후반 guidance 샤프닝" |
| ③ | **Substructure Conditioner** | `density_loss_function.py:242-251` | `x_0_hat[~ROI_mask] = reference_coords` — ROI 밖을 참조좌표로 고정, ROI만 evidence 만족하게 이동 |
| ④ | **합성 Loss 추상화** | `abstract_loss_funciton.py`, `multi_loss_function.py` | `__call__(x_0_hat, t, ...) -> (loss, new_x_0_hat)`; MultiLoss가 데이터항+정규화항 가중합 |
| ⑤ | **앙상블 + Pruning** | `density_omp.py`(OMP greedy), `density_occ.py`(occupancy+entropy) | N_sample 중 관측값 설명하는 최소 subset 선택 |
| ⑥ | **Relaxation + Validity** | `relaxation.py`(AMBER), `violation_loss_function.py`, `bond_length_loss_function.py` | 미분가능 clash/bond 패널티 + 사후 force-field relax |

**핵심 통찰**: 우리는 실험 데이터가 없으므로 그들의 데이터항(density/NOE/ESP)은 못 쓴다.
대신 **"참조 촉매기하 loss"**를 데이터항 자리에 끼운다 — `DensityGuidanceLossFunction`을
`CatalyticGeometryLossFunction`으로 교체. 나머지 5개 부품(①②③⑤⑥)은 원리 그대로 이식.

---

## 3. 우리 시스템 매핑 — 무엇을 무엇으로 대체하는가

| v3 현재 (수작업) | guided_alphafold 부품 | V4 재구축 후 |
|---|---|---|
| `md/anchored_build.py` (WT backbone+pose graft) | ③ Substructure Conditioner | ROI 밖 = WT 촉매 scaffold 고정 (동일 원리, diffusion 내부화) |
| `md/pose_gate.py` (reference_like/alt/displaced 규칙) | ⑤ 앙상블 분포 readout | 규칙 대신 **앙상블에서 촉매적합 geometry 유지 비율/분산**으로 정량화 |
| per-mutant Boltz 단일 pose (8.3 Å 노이즈) | ①②⑥ N_sample guided 앙상블 | 붕괴된 1개 pose → 참조-guided 분포 |
| `md/nac.py` NAC (MD 후 사후계산, hard cutoff) | ④ **catalytic-geometry loss로 승격** + soft kernel | NAC 기하(donor-acceptor 거리/각)를 guidance loss로 pose 생성단에 주입 + reference 분포 대비 soft fit |
| `ranking/claim_guard` geometry-sparse 플래그 | ⑤ + evidence card | guided 앙상블 = geometry evidence 밀도 자체가 상승, card로 기록 |
| ML `ml_score` (activity-ish) | evidence prior (분리) | catalytic predictor 아님 → 축별 evidence prior로 분해 |

---

## 4. 목표 파이프라인 (개념)

```
sequence + ligand + mechanism
        │
homolog / MSA / initial complex
        │
active-state reference ensemble      ← 신규 (s04x)
        │
candidate mutation generation
        │
non-MD evidence prior (ML=prior)
        │
reference-state accommodation        ← anchored MD의 개명·역할축소
        │
optional MD / QM/MM / wet-lab triage
        │
ClaimGuard (최종 claim 제한)
```

현재 vs 개선:

```
현재:  WT 단일 pose → docking/ML/MD/NAC 판정
개선:  WT/homolog/experimental guided ensemble
       → reference confidence + geometry distribution
       → candidate accommodation / uncertainty scoring
       → ML은 prior, 최종 claim은 ClaimGuard로 제한
```

---

## 5. 아키텍처 — 모듈 레이아웃

### 5.1 신규 파이프라인 단계

**`s04x_reference_ensemble`** (s04와 s05 사이 삽입)

입력: 실험 구조, homolog ternary complex, constrained docking,
QM/MM-lite 또는 geometry-restraint minimization, (가능하면) guided-AF식 measurement-guided ensemble.

출력 (RunContext에 저장):
- `reference_conformers` — active-like 상태들의 집합 (단일 구조 아님)
- `reference_confidence` — ensemble 신뢰도 카드
- `nadp_anchor_distributions` — NADP atom-level anchor 거리/각 분포
- `formate_reactive_distributions` — formate reactive-position 분포
- `measurement_fit_metrics` — evidence 대비 fit
- `structure_validity_metrics` — clash/bond/strain

> "이 구조가 맞다"가 아니라 **"가능한 active-like 상태들의 분포"**를 저장한다.

> ⚠️ **resume 규약**: `s04x.load()`는 `s04x.run()`이 하는 모든 `ctx.put()`을 복원해야 한다
> (resume-load-parity 메모리 참조). 새 ctx 키 추가 시 run/load diff 필수.

### 5.2 소스 모듈 (`src/evoliez/guided/`)

```
guided/
  __init__.py
  adapter.py          ← GuidedReferenceAdapter (thin external adapter; vendor 금지)
  sampler.py          ← ① 루프 (Boltz denoiser 래핑; experiment_manager 이식)
  guidance_inject.py  ← ② grad 정규화 + step (get_x_t_from_x_0_hat 이식)
  conditioner.py      ← ③ substructure freeze (anchored_build 로직 재사용)
  prune.py            ← ⑤ occupancy/OMP (density_occ 이식; target = 촉매기하)
  ensemble_readout.py ← 앙상블 → 촉매적합 비율/분산 (pose_gate 대체)
  losses/
    base.py           ← ④ AbstractLoss + MultiLoss (clean-room 재작성)
    catalytic_geom.py ← NAC 기하 loss, 미분가능 (nac.py 재작성) ★신규 핵심
    reference_pose.py ← WT pose로의 2차 패널티 (eq.16 substructure conditioner)
    violation.py      ← ⑥ clash/bond validity (clean-room)
```

### 5.3 스키마 재구축 — FDH-일반화 우선

FDH에 특화하지 말고 **일반 효소용 schema를 먼저** 만든 뒤 FDH를 하나의 template으로 얹는다.

- `MechanismSpec` (v3 존재 → 확장)
- `LigandRole` (v3 존재)
- `ReactiveAtomMap` — reactive atom 쌍/각 정의 (SMARTS 기반, v3 nac에서 승격)
- `ReferenceEnsemble` — **신규**: conformers + confidence + 분포 + validity
- `EvidenceCard` (v3 존재 → reference-fit 축 추가)
- `ClaimGuard` (v3 존재 → ensemble-evidence 반영)

---

## 6. 핵심 설계 결정

### 6.1 NAC: binary → soft geometry score
WT에서도 NAC가 0이면 현재 cutoff는 논문용 label로 부적합.
V4: reference ensemble에서 C4-formate 거리·hydride angle·catalytic contact **분포**를 만들고,
후보가 그 분포를 얼마나 수용하는지 **soft kernel**로 평가.
(hard `NAC fraction cutoff` 폐기.)

### 6.2 anchored MD: 개명 + 역할 축소
`pose preservation validation` → **`reference-state accommodation / local relaxation filter`**.
"기능 보존 증명"이 아니라 **"이 후보가 reference ternary complex를 즉시 망가뜨리지 않는가"**만 본다.

### 6.3 de novo docking: hard label → uncertainty signal
Boltz/DiffDock/GNINA를 `displaced = inactive`로 쓰지 않는다.
여러 seed·여러 engine·reference-conditioned redocking consensus를 모아 **`pose_uncertainty`**로만 사용.

### 6.4 ML: catalytic predictor → evidence prior (축 분리)
ML 출력을 `activity improved` 단일 스코어가 아니라 다음으로 분리:
- stability prior
- evolution tolerance
- NADP-pocket compatibility
- formate positioning compatibility
- reference-state accommodation
- uncertainty / false-negative risk

최종 선별은 weighted sum보다 **Pareto / multi-lane selection**이 안전.

### 6.5 백엔드 — Boltz 이식 (Path B, 권장)
- **Path A (포크/vendor)**: guided_alphafold 직접 사용 → Protenix 백엔드.
  ✗ `inference.py`가 **CC-BY-NC 4.0(비상업)**, 내부 Protenix 파일에 출처/비상업 조건 혼재.
  ✗ 우리 Boltz 파이프라인과 이중 모델.
- **Path B (Boltz 이식, 권장)**: guidance 머신을 **우리 Boltz(MIT)에 clean-room 재작성**.
  Boltz도 diffusion sampler라 ①②③ 그대로 이식. 단일 모델·라이선스 클린·s04/s08b 통합.
- `guided_alphafold`는 **external dependency로만 평가**, 내부엔 `GuidedReferenceAdapter`
  얇은 어댑터만 둔다.

---

## 7. 실행 계획 (로컬-우선 원칙 준수)

| Phase | 내용 | 환경 | 목표/성공기준 |
|---|---|---|---|
| **0** | ④ loss 추상화 + `catalytic_geom.py`(미분가능 NAC) + `reference_pose.py` | 로컬 (`.venv-light`, 무 GPU) | 기존 s10 궤적 스냅샷에 대해 gradient가 촉매기하를 개선 방향으로 움직이는지 수치 검증 |
| **1** | Boltz sampler에 ①②③ 수술 → lead **mut_00167** 1개로 guided N=16 앙상블 | 서버 (소규모 GPU) | NADP 8.3 Å 노이즈가 참조-guided로 감소, 촉매적합 비율 산출. **GPU 메모리 실측 선행** |
| **2** | ⑤ pruning + ⑥ relax → 앙상블 확정, `pose_gate` 규칙을 readout으로 교체 | 서버 | measurement-consistent subset + validity 통과 |
| **3** | `s04x_reference_ensemble` 단계 wiring, ClaimGuard geometry-sparse 게이트를 ensemble-evidence로 갱신 | 서버 | end-to-end FDH 1 run |
| **4** | 스키마 일반화(효소-agnostic) + FDH를 template으로 재정의 | 로컬 | 비-FDH 2nd target smoke |

**즉시 착수 가능**: Phase 0. 서버 없이 기존 궤적으로 검증 가능한 유일 부분이자 재구축 성패의 핵심.

---

## 8. 리스크 & 가드레일

1. **미분가능 Boltz 디노이저 접근**이 관문. `torch.enable_grad`로 denoiser 통과 후
   `x.grad` 추출 → 순수 PyTorch라 가능하나 checkpointing/mixed-precision 끄고
   N=16 백프롭 메모리 감당 필요. **GPU 메모리 실측 선행 (Phase 1 진입 전).**
2. **여전히 thermodynamic population 아님** — guided 앙상블은 "촉매기하 정합 가설 분포"일 뿐.
   **활성 순위는 여전히 MD/RBFE 몫.** V4는 *더 나은 pose 생성기*이지 촉매력 추정기가 아니다.
   ClaimGuard·리포트에 반드시 반영.
3. **참조 loss 순환성**: WT 촉매기하로 너무 강하게 당기면 "변이가 WT처럼 보이게" 강제 →
   판별력 소실. `denoiser_time_index`(늦게 시작) + `step_size` 캘리브레이션으로
   prior(변이 효과) vs guidance(참조) 균형 필수 (논문 η 튜닝과 동일 문제).
4. **라이선스**: Path A는 CC-BY-NC 오염. Path B는 참조만 하고 Boltz 위에 재작성 → 코드 MIT 유지.
   `guided_alphafold` vendor 금지, adapter 경유만.
5. **v3 마감 보호**: V4는 별도 트랙. 진행 중인 v3 논문 본문에 끼워넣지 않는다.
   ctx.setup 규약(config 변경 시 purge) 위반 금지 — 라이브 run 건드릴 땐 `_state.json`
   fingerprint patch 선행.

---

## 9. 논문 claim 규율 (그대로 유지)

- ✗ "activity prediction", "functional-state proof"
- ✓ **"mechanism-informed candidate triage using active-state ensemble evidence"**
- ✓ 생성 ensemble = **"measurement/reference-consistent structural hypothesis"**
  (posterior structural hypothesis, NOT thermodynamic population)

이 논문이 우리에게 주는 것은 **"MD 없이 activity를 맞히는 법"이 아니라, reference state와
구조 불확실성을 더 과학적으로 다루는 법**이다.

---

## 10. 참고

- guided_alphafold GitHub: <https://github.com/sai-advaith/guided_alphafold>
- Nature Biotechnology paper: <https://www.nature.com/articles/s41587-026-03166-5>
- 코드 조사 대상 파일 (clean-room 참고): `experiment_manager.py`,
  `src/utils/non_diffusion_model_manager.py`, `src/losses/*`, `src/metrics/density_{omp,occ}.py`.
