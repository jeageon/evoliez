"""Configuration schema (spec section 19).

Pydantic models describing a full pipeline run. Loaded from YAML; unknown keys
are rejected so typos fail fast rather than silently doing nothing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

from evoliez.mechanism.spec import MechanismSpec


class Backend(str, Enum):
    """Adapter backend. ``mock`` runs everywhere; ``real`` shells out to the
    actual tools on the GPU server."""

    mock = "mock"
    real = "real"


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class LigandInput(_Base):
    id: str = "ligand_X"
    type: str = Field("smiles", description="smiles | sdf | mol2 | pdb | inchi | ccd")
    value: str = Field(..., description="SMILES string or path to a structure file")
    # Role-based docking manifest (GENERIC — never hardcode a ligand's identity).
    # `role` is free-form (e.g. design_ligand | cofactor | substrate | product |
    # ion | other). `role=None` means the POSITIONAL default: the primary
    # `InputConfig.ligand` is the design ligand, each `extra_ligand` a cofactor.
    # `dock` / `keep_as_context` override the role-derived defaults when set:
    #   dock            None -> True iff resolved role == "design_ligand"
    #   keep_as_context None -> True iff resolved role != "design_ligand"
    role: Optional[str] = None
    dock: Optional[bool] = None
    keep_as_context: Optional[bool] = None
    # Charge / parameterization (cofactor microspecies safety). `net_charge` is the
    # design-ligand TOTAL charge for MD/parameterization. NADP+ is net -3 at pH 7 --
    # the "+" in the name is the nicotinamide pyridinium REDOX state, NOT the
    # molecular charge (RCSB CCD `NAP` is the neutral 76-atom microspecies; the
    # physiological -3 species is 73 atoms). Leave None to infer from the structure.
    # For a large charged cofactor AM1-BCC/sqm does NOT converge, so supply FIXED
    # charges via `charges_mol2` (a pre-charged GAFF mol2) and set `allow_am1bcc:
    # false` -- the pipeline then NEVER falls back to the failing on-the-fly charge
    # derivation and fails LOUDLY if no fixed charges are available.
    net_charge: Optional[int] = None
    charges_mol2: Optional[str] = None      # path to a pre-charged GAFF mol2 (skips AM1-BCC)
    allow_am1bcc: bool = True               # set false for NADP/NADPH/ATP-class cofactors


class InputConfig(_Base):
    target_id: str = "target"
    target_fasta: Optional[str] = None
    target_sequence: Optional[str] = None
    ligand: LigandInput
    # Additional ligands (cofactors / substrates) co-modelled in the s04 Boltz
    # complex as their OWN entities, for a complete active site. The design
    # objective still targets `ligand` (the affinity binder); these are
    # structural context, not the optimisation target.
    extra_ligands: List[LigandInput] = Field(default_factory=list)

    # Optional structural / biochemical context (spec 4.2).
    target_structure: Optional[str] = None
    catalytic_residues: List[str] = Field(default_factory=list)
    fixed_residues: List[str] = Field(default_factory=list)
    known_binding_site: List[str] = Field(default_factory=list)
    cofactor: Optional[str] = None
    target_ph: float = 7.4
    organism: Optional[str] = None
    ec_number: Optional[str] = None
    # Provenance accessions for the methods section (s01 report surfaces these as
    # structured metadata). `accession` = the UniProt accession the sequence /
    # residue numbering is taken from; `pdb_id` = the reference PDB (when a
    # structure underlies the numbering); `numbering_scheme` = how residue indices
    # are numbered (e.g. "UniProt canonical", "PDB 3JTM author numbering",
    # "as-provided"). All optional — absent => the report keeps its "not provided /
    # verify" warning, so existing configs render unchanged.
    accession: Optional[str] = None
    pdb_id: Optional[str] = None
    numbering_scheme: Optional[str] = None
    experimental_dataset: Optional[str] = None

    @model_validator(mode="after")
    def _need_sequence_source(self) -> "InputConfig":
        if not self.target_fasta and not self.target_sequence:
            raise ValueError("input requires either target_fasta or target_sequence")
        return self


# --------------------------------------------------------------------------- #
# Stage configs
# --------------------------------------------------------------------------- #
class HomologConfig(_Base):
    method: str = "mmseqs2"  # blastp | jackhmmer | mmseqs2
    database: Optional[str] = None
    # mmseqs GPU search: `database` must be a makepaddedseqdb-padded DB and a free
    # GPU pinned via CUDA_VISIBLE_DEVICES -> fast local search, no 16h CPU index.
    mmseqs_gpu: bool = False
    # Per-tool database overrides for first-class multi-tool sources (s02
    # gather_homologs). When `sources` names a CONCRETE tool (mmseqs2 / jackhmmer
    # / blastp / hhblits), it searches databases[tool] (falling back to
    # `database`), so several sequence tools run as INDEPENDENT tracks against
    # their own DB format (mmseqs padded seqdb / FASTA / HH-suite ffindex) and
    # merge — each tool finds a partly-distinct homolog pool (user §7).
    databases: Dict[str, str] = Field(default_factory=dict)
    # jackhmmer / HHblits binaries can live in a DEDICATED conda env (off the
    # pipeline PATH — e.g. when the production env's HMMER is missing its BLAS
    # dep): point *_bin at them and the runner prepends that env's lib to
    # LD_LIBRARY_PATH so the tool's shared libs (libopenblas, …) load. HHblits
    # (HH-suite profile–profile HMM) is the most sensitive remote-homolog track;
    # hhblits_iterations = `hhblits -n`.
    jackhmmer_bin: Optional[str] = None
    hhblits_bin: Optional[str] = None
    # Full path to the mmseqs binary when it isn't on PATH (e.g. it lives in a
    # different conda env than the pipeline). Used by the local-MSA path too.
    mmseqs_bin: Optional[str] = None
    hhblits_iterations: int = 2
    # jackhmmer is bottlenecked by a SINGLE DB-reader/dispatch thread (~2 MB/s),
    # so --cpu barely helps. >1 here splits the target DB into N chunks and runs
    # one jackhmmer per chunk IN PARALLEL (chunks staged in /dev/shm to dodge a
    # fragmented HDD), bypassing that bottleneck for ~N-fold speedup on N cores.
    jackhmmer_chunks: int = 1
    # Thread cap for the CPU-only sequence tools (jackhmmer / hhblits): keep
    # modest on the shared server so a homolog search never monopolises cores.
    search_threads: int = 4
    # Per-track homolog cache. When set, each source's result is cached HERE
    # (outside the run dir) under a content key = hash(query + that track's DB +
    # its search params), so a finished track is reused across runs even when an
    # UNRELATED config field changes (which otherwise purges the run dir). Lets
    # us nail one track at a time without re-running the others (user §7). Unset
    # -> cache lives in the run dir (lost on a config-fingerprint purge).
    cache_dir: Optional[str] = None
    max_sequences: int = 5000
    identity_min: float = 0.20
    identity_max: float = 0.95
    coverage_min: float = 0.70
    evalue_max: float = 1e-5
    length_ratio_min: float = 0.7
    length_ratio_max: float = 1.3
    cluster_identity: float = 0.90
    # Integrated multi-source homology: merge any of these (s02 gather_homologs).
    #   sequence  -> local mmseqs/jackhmmer/blastp DB, OR ColabFold remote MSA
    #   structure -> Foldseek (ProstT5 seq->3Di) against a 3Di structure DB
    sources: List[str] = Field(default_factory=lambda: ["sequence"])
    use_foldseek: bool = False          # legacy alias: true => adds 'structure'
    foldseek_database: Optional[str] = None   # Foldseek 3Di DB (AFDB50/PDB100)
    foldseek_prostt5: Optional[str] = None    # ProstT5 weights: predict 3Di from
    # the query SEQUENCE (no structure needed). `foldseek databases ProstT5 <dir>`
    foldseek_max_seqs: int = 2000


class MSAConfig(_Base):
    method: str = "mafft"  # mafft | mmseqs2
    compute_conservation: bool = True
    compute_pssm: bool = True
    compute_covariation: bool = False
    max_gap_frequency: float = 0.5
    remote_server: bool = False  # use a hosted MSA API instead of local DBs
    # ESM2 single-sequence prior (s03): per-position substitution variability
    # from a protein language model — MSA-free, complements the MSA columns.
    esm_enabled: bool = False
    esm_model: str = "esm2_t33_650M_UR50D"


class ComplexPredictionConfig(_Base):
    primary_method: str = "boltz2"  # boltz2 | boltz1
    use_msa_server: bool = True
    representative_homologs: int = 20
    use_templates: bool = True
    pocket_constraints: bool = True
    predict_affinity: bool = True
    diffusion_samples: int = 5  # Boltz poses per prediction (ensemble features)
    # Boltz-2's optimized triangular kernels need the optional NVIDIA
    # `cuequivariance_torch` dep and HARD-FAIL (ModuleNotFoundError) if it's
    # absent rather than falling back to pure torch. Default OFF (-> pass
    # --no_kernels) so a stock `pip install boltz` env works; flip on only
    # after installing cuequivariance-torch (faster, GPU-specific).
    use_kernels: bool = False


class DockingConfig(_Base):
    methods: List[str] = Field(default_factory=lambda: ["vina"])  # vina|gnina|diffdock
    poses_per_candidate: int = 10
    pose_rmsd_cluster: float = 2.0
    # AutoDock Vina is O(exhaustiveness x ligand flexibility). A big flexible
    # cofactor like NADP (44 heavy, ~11 rotatable) at the default
    # exhaustiveness=8 with no time bound can run for hours and, with
    # captured stdout, *looks* hung. Bound it; smoke configs drop it further.
    exhaustiveness: int = 8
    cpu: int = 0          # 0 = Vina default (all cores); >0 caps it (shared box)
    timeout_s: int = 1800  # hard wall so a runaway dock fails loudly, never hangs


class StabilityConfig(_Base):
    method: str = "foldx"  # foldx | rosetta | thermompnn | ml
    max_ddg_allowed: float = 2.5


class ReactiveGeometryConfig(_Base):
    """Near-attack-conformation (NAC) / catalytic-power screen on the s10
    trajectory (see evoliez.md.nac). md_lite_score only proves BINDING/structural
    stability; this adds REACTIVITY -- the fraction of frames where the reacting
    atoms sit in a productive geometry (transferring atom within distance_max of
    the acceptor AND a donor-transfer-acceptor angle >= angle_min). Disabled by
    default; enzyme-specific SMARTS + criteria live in the target config (this is
    generic). FDH example: donor formate '[CX3H1](=O)[O-]', acceptor the
    nicotinamide C4 '[cH1]([cH0]C(=O)[NX3])[cH1]'."""
    enabled: bool = False
    donor_smarts: str = ""
    acceptor_smarts: str = ""
    # PROTEIN-nucleophile donor (serine hydrolase / protease): 'RESNAME:ATOM[:RESNUM]'
    # (e.g. 'SER:OG:68'). When set, the nucleophile is this PROTEIN catalytic-residue atom
    # (resolved from the MD topology) instead of a ligand ``donor_smarts`` match; the ligand
    # then supplies only the acceptor (scissile carbonyl C). Generic serine/cysteine-hydrolase
    # feature -- see docs/car_v5/nonredox_tem1_plan.md.
    donor_protein: "Optional[str]" = None
    donor_idx: int = 0
    acceptor_idx: int = 0
    transfer_is_h: bool = True
    distance_max: float = 3.5     # Å, transferring atom -> acceptor
    angle_min: float = 150.0      # deg, donor_heavy - transfer - acceptor
    label: str = "reaction"
    # NAC validity gates (decouple placement / retention / reactive geometry, so a
    # diffused or mis-placed co-substrate is reported INVALID, not "low reactivity").
    placement_distance_max: float = 4.0   # Å, frame-0 transfer->acceptor (Michaelis-like start)
    retention_distance_max: float = 6.0   # Å, "still in the active-site pocket" cutoff
    retention_min_fraction: float = 0.8   # require this fraction retained for a valid NAC
    # Co-substrate retention restraint (NAC-3): keep a FREE co-substrate (e.g. formate)
    # from diffusing out of the pocket in implicit solvent. A flat-bottom restraint on
    # the formate-COM -> acceptor DISTANCE only -- it must NOT restrain the reactive
    # angle (that would manufacture NAC). Off by default; when on the NAC is labelled
    # valid_restrained_retention_screen and the restraint energy is reported.
    restrain_cosubstrate: bool = False
    # NAC-4: before the MD, REPLACE the predictor's (often random) co-substrate pose with
    # the constructed near-attack geometry off the acceptor face, so a mis-placed formate
    # is not wrongly scored / skipped. The MD + retention restraint then test whether the
    # active site MAINTAINS that reactive arrangement.
    template_cosubstrate_placement: bool = False
    restraint_radius_A: float = 5.0       # flat-bottom radius, formate COM -> acceptor atom
    restraint_k: float = 2.0              # kcal/mol/Å² harmonic beyond the flat well


class RBFEConfig(_Base):
    """Alchemical relative binding free energy (ΔΔG_bind, mutant vs WT) for the design
    ligand via softcore TI (adapters/amber_rbfe). FINAL confirmatory tier -- EXPENSIVE
    (2 legs x n_lambda windows x min+heat+prod PER mutated residue), so opt-in and run
    on the top_n MD candidates only. Needs Amber pmemd.cuda + explicit solvent.
    Multi-point candidates use the additive single-residue approximation (flagged)."""
    enabled: bool = False
    top_n: int = 3                  # run RBFE on the top-N MD candidates by md_lite_score
    n_lambda: int = 9              # Gauss-Legendre λ windows per leg
    min_cyc: int = 2000
    heat_steps: int = 10000
    prod_steps: int = 50000        # 50 ps/window at dt=0.001 ps
    multipoint: str = "additive"   # additive (sum per-residue) | skip


class BindingDGConfig(_Base):
    """Endpoint binding free energy via the Amber MM-PB/GBSA tier.

    OpenMM deliberately leaves ``binding_dg`` empty; when enabled, s10 runs a
    separate explicit-solvent Amber/MMPBSA confirmation on the top MD candidates
    and copies only the endpoint ΔG values back into the MD report/provenance."""
    enabled: bool = False
    top_n: int = 3


class MDConfig(_Base):
    enabled: bool = True
    engine: str = "openmm"
    reactive_geometry: ReactiveGeometryConfig = Field(
        default_factory=ReactiveGeometryConfig)
    rbfe: RBFEConfig = Field(default_factory=RBFEConfig)
    binding_dg: BindingDGConfig = Field(default_factory=BindingDGConfig)
    protocol_level: int = Field(1, ge=0, le=3)  # spec 15.2
    # implicit (GBSA/obc2) is the default. "explicit" (ROADMAP_V5 E1) is REAL on the multi-ligand
    # catalytic path: TIP3P water box + PME + Na+/Cl- neutralize (addSolvent before create_system),
    # so a charged substrate/cofactor/metal cluster is stabilised instead of diffusing out of a GBSA
    # pocket. Explicit on the single-ligand path still warns + falls back to implicit.
    solvent: str = "implicit"  # implicit | explicit (explicit needs the multi-ligand build)
    temperature_K: float = 300.0
    timestep_fs: float = 2.0
    minimize_steps: int = 5000
    equilibration_ps: float = 100.0
    restrained_md_ps: float = 500.0
    production_ns: float = 1.0
    replicas: int = 1
    top_candidates: int = 30
    # Paper-grade: send to MD ONLY candidates with a REAL s08b Boltz mutant complex
    # (boltz_delta_source=="real"), never a WT-coords identity-swap proxy. Falls back
    # to the full kept set only when NOTHING was folded (configs without s08b).
    require_real_structure: bool = True
    # Functional-state anchored validation: build each mutant from the REFERENCE
    # complex (reference backbone + reference cofactor/substrate poses + only the
    # point mutation, via md.anchored_build) instead of a fresh per-mutant Boltz
    # pose. Separates the mutation effect from pose-search noise; the s08b Boltz
    # pose is kept as an alternative-pose HYPOTHESIS, not the validated structure.
    # Falls back to the Boltz/proxy structure when no reference PDB is on disk.
    anchored_validation: bool = True
    # Reference-like pose gate (md.pose_gate): per-ligand pocket-aligned RMSD +
    # internal-shape RMSD vs the reference; tags reference_like / alternative_pose /
    # displaced. A non-reference-like cofactor pose is never silently an "improved
    # mutant".
    pose_gate_enabled: bool = True


class ValidationConfig(_Base):
    redocking: DockingConfig = Field(default_factory=DockingConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    md: MDConfig = Field(default_factory=MDConfig)


class MutationGenConfig(_Base):
    methods: List[str] = Field(
        default_factory=lambda: ["chemistry_rules", "msa_sampler"]
    )  # ligandmpnn | msa_sampler | chemistry_rules
    max_candidates: int = 2000
    design_radius_angstrom: float = 8.0
    # Whether the design mask includes the sphere around EVERY co-modelled ligand
    # (True, v2 default) or ONLY the primary design ligand + catalytic neighborhood
    # (False). Set False when cofactors sit in a DIFFERENT site/domain you want to
    # keep intact (e.g. CAR: design the A-domain 3-HP pocket, not the R-domain NADPH
    # pocket) so the design/MD budget isn't spent on off-target cofactor-pocket mutants.
    design_around_extra_ligands: bool = True
    fix_catalytic_residues: bool = True
    fix_highly_conserved_residues: bool = True
    conservation_fix_threshold: float = 0.9
    ligandmpnn_samples: int = 32
    ligandmpnn_temperature: float = 0.1
    # --- generation strategy (s07) ------------------------------------------ #
    # Per-generator share of max_candidates so the FIRST generator can't fill the
    # cap and starve the rest. Empty -> a sensible default that favours
    # LigandMPNN + the multipoint library. Keys: chemistry_rules | msa_sampler |
    # ligandmpnn | multipoint.
    generator_quota: Dict[str, float] = Field(default_factory=dict)
    # FuncLib-style multi-point active-site library: combine the allowed single
    # substitutions (from all generators) into 2..N-point combinations on
    # spatially-central (closest-to-ligand) designable positions, where active-
    # site epistasis lives. Sampled (bounded), not exhaustively enumerated.
    multipoint: bool = True
    multipoint_order: int = 3            # max simultaneous substitutions
    multipoint_max: int = 0             # cap (0 -> ~30% of max_candidates)
    # MSA-sampler generation GATES (not just a downstream score): drop columns
    # gappier than msa_gap_max; require the family actually tolerates the residue.
    msa_gap_max: float = 0.5
    msa_top_k: int = 4
    # Safety re-check on LigandMPNN designs (designable-only; never catalytic /
    # fixed; cap mutations per design so it stays a focused active-site edit).
    ligandmpnn_max_mut_per_design: int = 8
    # --- safety filters (s07) ----------------------------------------------- #
    # FLAG vs DROP risk-flagged candidates (introduce Gly/Pro/Cys; radical
    # physicochemical substitution in a catalytic residue's 2-shell). Default
    # False = flag-only (keep everything in ctx `candidates`, record the flag in
    # the generated-provenance table; downstream prunes). True drops the flagged
    # candidates from the shipped set (still recorded in provenance for audit).
    risk_filter: bool = False
    # --- budget tiers (s08-s10 cost; reviewer-facing) ----------------------- #
    # Tier the generated pool into single / multipoint / risky budget VIEWS, each
    # top-N by the existing per-candidate score, written to separate provenance
    # files (generated_candidates_{single,multipoint,risky}.csv). These do NOT
    # change ctx `candidates` (the full set) — downstream still prunes — they are
    # the cost-aware shortlist a reviewer reads.
    tier_single_top: int = 100
    tier_multipoint_top: int = 30
    tier_risky_top: int = 20


class RerankConfig(_Base):
    model: str = "xgboost"  # xgboost | mlp | rules
    use_experimental_labels: bool = False
    top_for_redocking: int = 200
    top_for_md: int = 30
    # explicit staged Boltz Δ (expert review #2): proxy Δ in the fast pass
    # (s08), then a real per-mutant Boltz re-evaluation on the top-N only.
    mutant_boltz_enabled: bool = True
    mutant_boltz_top_n: int = 100
    mutant_boltz_diffusion_samples: int = 3


class InteractionModelConfig(_Base):
    """Self-supervised family interaction-geometry model (spec 9.2 + 13.3).

    For each representative homolog: predict its structure, dock the ligand as
    a pose ensemble, extract per-ligand-atom interaction-distance descriptors.
    Statistically select family-consensus poses (positives) vs outliers/decoys
    (negatives); train a classifier; score mutant complexes by family
    consistency. No experimental labels required.
    """

    enabled: bool = True
    # How many subfamily representatives seed the Boltz family ensemble.
    #   <int> -> exactly that many (one per LARGEST cluster, topped up by
    #            identity); -1 -> every cluster.
    #   "auto" -> size it from the MSA itself: take the largest subfamily
    #            clusters until they cover `representative_coverage` of the
    #            homolog pool, clamped to [representative_min, representative_max].
    #            A deeper / more diverse MSA then trains on MORE representatives
    #            and a shallow one on fewer, instead of a hard-coded count.
    representative_homologs: Union[int, Literal["auto"]] = 24
    representative_coverage: float = Field(
        0.9, gt=0.0, le=1.0,
        description="auto: fraction of homologs to cover by descending cluster size",
    )
    representative_min: int = Field(20, ge=1)   # auto floor (data-size guard)
    representative_max: int = Field(150, ge=1)  # auto ceiling (compute cap)
    poses_per_homolog: int = 12
    docking_method: str = "vina"  # vina | gnina | diffdock
    contact_cutoff: float = 6.0  # ligand-atom -> residue interaction distance
    k_nearest_residues: int = 6  # per-ligand-atom nearest enzyme points
    # Where each representative's Boltz MSA comes from:
    #   "server" -> remote ColabFold (--use_msa_server): accurate but ~minutes/seq
    #               and N external requests (infeasible for a large ensemble).
    #   "local"  -> ONE batched mmseqs GPU search vs the local UniRef30 DB
    #               (homologs.databases["mmseqs2"] / homologs.database), split into
    #               per-rep a3m. Deep MSA, no external load — best for many reps.
    #   "single" -> single-sequence Boltz (msa: empty): fastest, lowest accuracy.
    rep_msa: Literal["server", "local", "single"] = "server"
    rep_msa_max_seqs: int = 2000  # local: per-rep MSA depth cap (mmseqs --max-seqs)
    # statistical pose selection
    pose_select_mad_z: float = 2.5  # keep poses within this robust z of consensus
    pose_outlier_mad_z: float = 4.0  # beyond this -> negative example
    min_decoys_per_homolog: int = 4  # synthetic negatives if poses too consistent
    # soft / multi-class labelling (expert review #3): mid-band poses are
    # "alternative/uncertain", NOT auto-negatives; add hard wrong-pose decoys;
    # report a subfamily-holdout AUROC.
    keep_alternative_band: bool = True
    alternative_weight: float = 0.3
    hard_decoys_per_homolog: int = 2
    subfamily_holdout: bool = True
    # --- multi-engine pose consensus (GNINA / DiffDock augmentation) --------- #
    # The Boltz family consensus is the POSITIVE TEACHER. Docking poses augment
    # the training data ONLY relative to that consensus: a docking pose that
    # AGREES (close RMSD + high contact-fingerprint overlap + no clash + key/
    # catalytic contacts preserved) is a WEAK POSITIVE; one that scores well yet
    # DISAGREES is a HARD NEGATIVE (the valuable signal); a clashing / ligand-
    # inverted / catalytic-broken pose is EXCLUDED. Default OFF -> behaviour is
    # byte-identical to a Boltz-only run.
    multi_engine: bool = False
    multi_engine_methods: List[str] = Field(default_factory=list)  # gnina|diffdock
    # Per-rep docking augmentation scope: 0 = ALL representatives (WT + every rep,
    # ~2xN docking jobs); N>0 = cap to an identity-stratified sample of N reps (the
    # WT is always docked). Keeps the augmentation from becoming a large standalone
    # docking stage when the representative count is big.
    multi_engine_max_reps: int = 0
    consensus_overlap_rmsd: float = 2.0   # RMSD-to-consensus-pose AGREE threshold (Å)
    consensus_overlap_fp: float = 0.6     # contact-fingerprint overlap AGREE threshold
    hard_negative_weight: float = 1.0     # sample weight for discordant-high-score rows
    weak_positive_weight: float = 0.3     # sample weight for agreeing docking rows
    # model
    model: str = "xgboost"  # xgboost | logistic | heuristic (auto-fallback)

    @model_validator(mode="after")
    def _check_representative_bounds(self) -> "InteractionModelConfig":
        if self.representative_min > self.representative_max:
            raise ValueError(
                f"representative_min ({self.representative_min}) must be <= "
                f"representative_max ({self.representative_max})"
            )
        return self


class AdvancedConfig(_Base):
    """Accuracy / paper-readiness layers (user guidance). All default on;
    each degrades gracefully and never produces a supervised label."""

    mechanism: bool = True
    mechanism_annotation_file: Optional[str] = None  # M-CSA-style JSON
    ligand_importance: bool = True
    interaction_fingerprint: bool = True
    negative_design: bool = True
    subfamily_msa: bool = True
    calibration: bool = True
    active_learning: bool = True
    provenance: bool = True
    al_beta: float = 0.3   # uncertainty (explore) weight
    # (diversity is enforced by the focused-library cluster round-robin, not a
    # score weight, so there is no al_gamma knob.)


class GNNConfig(_Base):
    """Server-grade EvoLigand-GNN (E(3)-invariant relative-vector model).

    Optional: torch is server-only. When disabled / no checkpoint / no torch,
    ranking falls back to the heuristic family interaction model. Train with
    ``evoliez train-gnn`` after ``evoliez build-graph-dataset``.
    """

    enabled: bool = False
    build_dataset: bool = True  # export graph dataset during the pipeline
    checkpoint: str = "checkpoints/evoligand_gnn.pt"
    hidden_dim: int = 128
    layers: int = 4
    rbf: int = 16
    equivariant: bool = False  # True = E(3)-equivariant coord-update layers
                               # (à la EGNN); default = E(3)-invariant readout
    radius_lr: float = 6.0  # ligand-atom -> residue edge radius (Å)
    radius_rr: float = 8.0  # residue-residue spatial edge radius (Å)
    lr: float = 1.0e-3
    epochs: int = 20
    amp: bool = True
    ddp: bool = False  # set when launched via torchrun on multi-GPU
    # confidence-aware knobs (user guidance): pLDDT/PAE/disorder as features +
    # edge weights + coordinate-noise augmentation, never flexibility labels.
    use_disorder: bool = True
    low_plddt_cutoff: float = 50.0       # below -> very-low confidence
    drop_far_low_plddt: bool = True      # drop only if also >10 Å from ligand
    coord_noise_min: float = 0.1         # Å jitter for high-pLDDT residues
    coord_noise_alpha: float = 1.5       # extra Å jitter scaled by (1 - pLDDT)


class OutputConfig(_Base):
    final_library_size: int = 96
    top_single_mutants: int = 50
    report_format: List[str] = Field(default_factory=lambda: ["markdown", "csv"])


class ScoreWeights(_Base):
    """Final multi-objective weights (spec section 16)."""

    ml_mutation: float = 1.0
    ligand_interaction_gain: float = 1.0
    msa_permissiveness: float = 1.0
    complex_confidence: float = 0.5
    redocking_consistency: float = 0.5
    key_contact_preservation: float = 1.0
    stability: float = 0.75
    md_lite: float = 1.0
    family_interaction: float = 1.0  # learned family-geometry consistency
    mutant_boltz_gain: float = 1.0  # real per-mutant ΔBoltz ligand-binding gain (s08b)
    gnn: float = 0.0  # EvoLigand-GNN score (0 unless a model is trained)
    # penalties
    conservation_penalty: float = 1.0
    catalytic_geometry_penalty: float = 1.5
    ddg_penalty: float = 0.75
    clash_penalty: float = 1.0
    docking_uncertainty_penalty: float = 0.5
    md_instability_penalty: float = 1.0
    # negative design (user §4): explicitly avoid risky mutations
    neg_catalytic_mut: float = 4.0
    neg_conserved_motif: float = 1.5
    neg_buried_core_polar: float = 1.0
    neg_catalytic_geometry: float = 2.0
    neg_overbinding: float = 0.75
    neg_pose_inversion: float = 1.0
    # mechanism / specificity bonuses
    catalytic_geometry_preservation: float = 1.0
    specificity_divergence_bonus: float = 0.5
    # Catalytic-power (NAC) bonus on ΔNAC vs WT (s10 reactive_geometry). 0.0 by
    # DEFAULT -- NAC stays a diagnostic and does NOT move the ranking unless a
    # cofactor-switching/activity objective explicitly opts in by raising this.
    catalytic_nac: float = 0.0


class ProjectConfig(_Base):
    name: str = "evoliez_project"
    objective: str = "binding_enhancement"
    output_dir: str = "runs/evoliez_project"


class SelectionLanesConfig(_Base):
    """Multi-lane candidate selection (ranking.multi_lane) — ML as a PRIOR, not a
    hard filter. When enabled, s09 builds the MD set as a UNION of lanes instead of
    the single _md_key cut, so an ML false negative in one lane is caught by another
    (incl. a low_ml_control probe). Opt-in: default off keeps the existing behaviour."""
    enabled: bool = False
    from_ml_high: int = 25
    from_stability_high: int = 12
    from_geometry_high: int = 12
    from_diversity: int = 8
    low_ml_controls: int = 8


class ReferenceEnsembleConfig(_Base):
    """ROADMAP_V3 B5 / V4 — optional active-state reference-ensemble stage (s04x),
    inserted after s04 complex prediction when ``enabled``. Default off keeps the
    s01-s11 pipeline unchanged. ``seed_manifest`` overrides the frozen v4 seed path
    (else the FDH default / the ``EVOLIEZ_SEED_MANIFEST`` env)."""
    enabled: bool = False
    seed_manifest: Optional[str] = None


class ComputeConfig(_Base):
    """GPU-first, CPU-bounded execution (ROADMAP_V2 §2b / Phase H1). `cpu_core_budget` caps
    OMP/MKL/OpenBLAS/NumExpr + every process pool so no stage trips the shared-server
    watchdog (~48 cores for 10 min); `gpu_pool` is the GPU set the GPU-bound stages
    (s04/s08b/s09/s06b/s10) fan across. Empty `gpu_pool` = inherit CUDA_VISIBLE_DEVICES.
    `fail_loud_on_cpu_md` makes s10 raise instead of silently running MD on CPU (~200x slow)."""
    cpu_core_budget: int = 16
    gpu_pool: List[int] = Field(default_factory=list)
    fail_loud_on_cpu_md: bool = True


class Config(_Base):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    input: InputConfig
    homologs: HomologConfig = Field(default_factory=HomologConfig)
    msa: MSAConfig = Field(default_factory=MSAConfig)
    complex_prediction: ComplexPredictionConfig = Field(
        default_factory=ComplexPredictionConfig
    )
    interaction_model: InteractionModelConfig = Field(
        default_factory=InteractionModelConfig
    )
    gnn: GNNConfig = Field(default_factory=GNNConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    mutation_generation: MutationGenConfig = Field(default_factory=MutationGenConfig)
    reranking: RerankConfig = Field(default_factory=RerankConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    scoring: ScoreWeights = Field(default_factory=ScoreWeights)
    selection_lanes: SelectionLanesConfig = Field(
        default_factory=SelectionLanesConfig)
    reference_ensemble: ReferenceEnsembleConfig = Field(
        default_factory=ReferenceEnsembleConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    # ROADMAP_V3 B4 — the mechanism envelope (reaction.class template + required
    # reaction_state + geometry terms). Optional and opt-in: when set, it is
    # hard-gate validated at config-load (a missing required reaction_state field
    # ABORTS) and the pipeline (s06) puts it + its runtime geometry terms into ctx
    # so NAC / s09 / s10 use mechanism-configured geometry instead of the legacy
    # features.mechanism heuristic. Absent = existing behaviour (legacy annotator).
    mechanism: Optional[MechanismSpec] = None

    # Global default backend and optional per-stage overrides
    # (keys = stage name, e.g. {"s04_complex": "real"}).
    backend: Backend = Backend.mock
    backends: Dict[str, Backend] = Field(default_factory=dict)

    # Production safety: under backend=real a missing/failed real tool (Boltz,
    # Vina, GNINA, DiffDock) or an unparseable ligand HARD-FAILS instead of
    # silently degrading to a mock/synthetic artifact (audit P0 #3). Set true to
    # permit that degradation (warns loudly) - e.g. partial-coverage smoke runs.
    allow_mock_fallback: bool = False

    seed: int = 1234

    def backend_for(self, stage_name: str) -> Backend:
        return self.backends.get(stage_name, self.backend)


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config, apply optional dotted-key overrides, validate."""
    path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(raw, dotted, value)
    return Config.model_validate(raw)


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
