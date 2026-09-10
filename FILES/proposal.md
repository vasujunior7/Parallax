# Parallax — An Active-Perception Inspection Agent That Knows When It Doesn't Know

**OpenCV AI Competition 2026, powered by AWS — AWS Cloud Compute Grant Proposal**

| | |
|---|---|
| **Team name** | **Parallax** |
| **Team representative** | **Aditya Kumar** — vasudeo118@gmail.com |
| **Featured paths pursued** | **Both** — Agentic Vision (primary) and Best Use of COOL (secondary) |
| **Submission target** | October 26, 2026, 11:59 p.m. Pacific Time |

---

## 1. Problem Statement and Intended Real-World Impact

Automated visual inspection does not fail in production because models cannot find defects. It fails because
models cannot tell you **when to stop trusting them**. A deployed inspection system meets lighting it was never
trained under, a part orientation nobody anticipated, a new supplier's surface finish — and it returns a
confident verdict anyway. The industry's response is to route everything to human review, which erases the
economics, or to route nothing, which ships defects.

The missing component is a *trustworthy, cheap, real-time signal for when a perception result is
out-of-distribution* — and an agent that acts on that signal instead of narrating it.

**Parallax** is a cloud inspection service built around that signal. OpenCV 5 performs the geometry and
metrology: camera calibration, learned feature matching and homography alignment to a golden reference,
differencing, morphological defect segmentation, and contour-based defect measurement. A frozen vision backbone
with a lightweight interpretability probe then scores how far each frame sits from the distribution the system
was validated on. **That score — not a language model's self-assessment — decides what the agent does next:**
accept and log the verdict, re-capture the scene differently and re-run the OpenCV pipeline, or stop and
escalate to a human with the visual evidence attached.

**Intended impact.** Manufacturing QA teams currently choose between full automation they cannot audit and
manual review they cannot afford. A calibrated escalation policy converts that binary into a dial: the operator
sets an accuracy floor, and the system reports what fraction of frames it can handle autonomously at that floor.
We will publish that trade-off curve as our headline result. The same architecture — perception, measured
confidence, bounded autonomy, human gate — transfers directly to medical imaging triage, infrastructure
inspection, and any safety-adjacent vision deployment where being wrong quietly is worse than being slow.

**Responsible-use position.** The system is designed to *reduce* unwarranted automation. It never hides an
uncertain result behind a confident-sounding summary, it blocks rather than guesses when the probe flags
out-of-distribution input, and every autonomous verdict is recorded with the evidence that produced it. We will
report our failure cases — including defects the probe was confidently wrong about — as a first-class section of
the technical report rather than an appendix.

---

## 2. Planned OpenCV 5 Image and Video Analysis

OpenCV 5 performs the substantive perception work. It is not preprocessing around a model call; the geometric
and metrological stages below have no learned substitute in our pipeline, and their outputs are what the agent
reasons over.

Module names below follow the **OpenCV 5 reorganization**, in which `calib3d` was split into `geometry`,
`calib`, and `stereo`, and `features2d` was renamed `features`.

| Stage | OpenCV 5 modules | Purpose |
|---|---|---|
| Calibration | `calib` — `calibrateCamera`; `imgproc` — `undistort` | Recover intrinsics on the live-capture path (see scale note below) |
| Feature extraction | `features` — ORB/SIFT, plus **ALIKED / DISK** learned local features | Detect correspondences robust to illumination change |
| Reference alignment | `features` — **LightGlue** matcher; `geometry` — `findHomography` with RANSAC; `imgproc` — `warpPerspective` | Register each captured frame to the golden reference for its part class |
| Change isolation | `core` — `absdiff`; `imgproc` — adaptive threshold, CLAHE normalisation | Isolate genuine surface deviation from lighting and pose variation |
| Defect segmentation | `imgproc` — `morphologyEx` open/close, `findContours`, `connectedComponentsWithStats` | Produce defect masks and per-blob geometry |
| Metrology | `imgproc` — `contourArea`, `minAreaRect`, `arcLength` | Convert blobs to physical measurements and apply pass/fail tolerances |
| Classification | `dnn` — ONNX inference on candidate regions | Assign defect type to each candidate region |
| Video path | `video` — `calcOpticalFlowFarneback`, background subtraction | Frame selection and motion-stability gating on continuous capture |

**Version compliance.** We build against OpenCV 5.x and print the runtime version banner in both the demo video
and the CI log. On the Arm path we use the Cloud Optimized OpenCV Library (COOL) AMI for Graviton4, which is
itself based on OpenCV 5. Our alignment stage uses **LightGlue with ALIKED/DISK features — capabilities
introduced in OpenCV 5 and unavailable in 4.x** — so the dependency on OpenCV 5 is functional rather than
nominal. We benchmark LightGlue against classical ORB + BFMatcher alignment and report both.

**Scale and measurement — what we can and cannot claim.** VisA ships no calibration targets or camera
intrinsics, so full metric calibration is not available on the dataset. On the VisA evaluation path we therefore
convert pixels to millimetres using a **known reference dimension of each part class** (a single scale factor
per class), and we report defect size in millimetres only under that stated assumption. Full intrinsic
calibration with `calibrateCamera` and `undistort` applies to the **live-capture judge-demonstration path**,
where we control the camera and can image a calibration target. We state this split explicitly rather than
implying metric accuracy we have not established.

**Dataset.** VisA (Visual Anomaly, Amazon Science) — 10,821 high-resolution images across 12 object classes with
pixel-level anomaly annotations, released under **CC BY 4.0**. We selected VisA specifically because its
permissive licence is compatible with the broad licence participants grant OpenCV and AWS over submitted
materials; we deliberately avoided MVTec AD, whose CC BY-NC-SA 4.0 terms conflict with that grant. All model
weights used are Apache 2.0 or MIT (see Section 7).

**Scoping the alignment pipeline to the classes it suits.** Homography registration against a golden reference
assumes a rigid, near-planar part. That holds for VisA's structured classes — `pcb1`–`pcb4`, and the fixtured
`capsules` and `candle` — and it does not hold for the deformable, pose-varying food classes (`cashew`,
`chewinggum`, `fryum`, `pipe_fryum`, `macaroni1`, `macaroni2`), where there is no meaningful golden reference to
difference against. **We therefore scope the align-and-difference pipeline to the rigid subset and treat the
deformable classes as a stated limitation**, handled by the `dnn` classification path and the confidence probe
alone, without reference differencing. If time permits, we will evaluate a deformable extension using
`estimateAffine2D` on local patches as a stretch goal and report it separately. We would rather define the
operating envelope honestly than report an average that hides a mode where the geometry is meaningless.

---

## 3. The Agentic Vision Loop

The qualifying requirement for the Agentic Vision path is that image or video results must change what the
system does next. In Parallax this is the central mechanism, not an added feature.

Each captured frame produces two independent products: an **OpenCV 5 verdict** (defect mask, measurements,
class) and a **confidence state** from the probe sidecar. The planner's action is a function of the confidence
state:

- **In-distribution and confident** -> accept the verdict, write it to the audit log, advance.
- **Low confidence or out-of-distribution** -> issue a *different* capture or parameter tool call — re-crop at
  higher resolution, change the viewing angle or illumination profile, adjust threshold and morphology
  parameters — then re-run the OpenCV 5 pipeline and re-score. The verdict can and does change.
- **Still uncertain after N re-looks** -> block, open a human approval task, and attach the visual evidence:
  which image patches drove the score, and which concept region they landed in.
- **Planner rollouts disagree** -> escalate. We sample a small number of planner rollouts and escalate when they
  fork rather than converge, an approach inspired by Goodfire's *Forking Fast* treatment of uncertainty dynamics
  (adapted as a heuristic; we do not claim to reimplement the paper's estimator).

Every transition is traced with the frame, the OpenCV outputs, the probe score, the planner's chosen action, and
the outcome — producing a replayable audit trail rather than an opaque decision.

**Autonomy is bounded by construction.** The agent may re-capture and re-parameterise freely; it may never
override a block, and it may never mark a part as passing while the probe reports out-of-distribution. Human
approval is a hard gate in the state machine, not a suggestion in a prompt.

---

## 4. The Confidence Sidecar (Goodfire-inspired)

The probe sidecar adapts Goodfire's production interpretability pattern — demonstrated with Rakuten, where SAE
probes on a frozen sidecar model generalised from synthetic training data to real production data, ran 10-500x
cheaper than LLM-as-judge setups at comparable accuracy, and where white-box probing of a model beat black-box
prompting of that *same* model by 96% to 51% F1 — from language to vision.

**Our adaptation.** A frozen **DINOv3 ViT-B/16** backbone (`facebook/dinov3-vitb16-pretrain-lvd1689m`) produces patch
activations for each aligned frame. This is the backbone the Block-Sparse Featurizer was developed and
validated against, so the published results transfer to our setting directly rather than by analogy.
On those activations we train:

1. A **linear probe** — our guaranteed floor, roughly one day of work.
2. A **Block-Sparse Featurizer** — our primary signal (Fel, Kowal, Jacobs, Hazra, Bhalla et al.,
   *Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds*, arXiv:2606.25234). We use
   the authors' reference implementation, <https://github.com/goodfire-ai/block-sparse-featurizer> (MIT),
   pinned at commit `219f121ea82d2b19200d1dac918396e6058d7eb9`, rather than reimplementing it. A BSF block yields two quantities where a
   sparse autoencoder yields one: the block norm, meaning *how strongly a concept is present*, and the block
   coordinate, meaning *where within that concept* the activation lies. For inspection this is the difference
   between "this resembles a weld seam" and "this is the cracked end of the weld-seam manifold" — an explanation
   we can render directly into the human escalation UI.

**What the repository ships, and what it does not.** It ships training code and three featurizer variants
(`VanillaBSF`, `GrassmannianBSF`, `GroupLassoBSF`) behind one `encode` / `loss` / decoder interface, with one
trainer and one visualiser; it does **not** ship pretrained featurizers, so we train ours and pin the commit so
a judge can reproduce it. Training is tractable within our window: the published quickstart trains on patch
activations from a few hundred images and runs on CPU, with the GPU needed only for the one-time activation
extraction pass.

**Matching the reference setting exactly.** We run the same backbone the BSF authors used, which removes a
class of avoidable risk. The repository's `bsf.data` helper loads `facebook/dinov3-vitb16-pretrain-lvd1689m`
at 224 px - 14x14 = 196 patch tokens at `d = 768`, dropping the first 5 tokens (CLS plus four registers) - and
we use it as shipped. Critically, `pos_mean.npy` (a 196x768 per-patch-position mean over roughly 25M ImageNet
patches, computed **for this backbone**) applies directly, so we inherit the authors' positional normalisation
instead of estimating our own from a smaller corpus. The normalisation convention is not optional: centre by
that mean, then scale so the mean squared activation norm equals `d`. The featurizers' sparsity thresholds
assume it.

**Licensing note.** DINOv3 is released under Meta's DINOv3 License rather than a permissive OSI licence, and
the weights are access-gated. We have accepted those terms deliberately, having weighed them against the
benefit of reproducing the reference setting exactly. The featurizer core consumes an `(N, d)` activation
matrix and is backbone-agnostic, so a swap to DINOv2 (Apache 2.0) remains a configuration change plus one
extraction pass should that trade-off need revisiting.

**Why the published results make this a targeted choice rather than a hopeful one.** The BSF paper's
minimum-description-length analysis finds all three variants describe activations more compactly than
direction-based featurizers, with recovered concepts typically **two- to four-dimensional** — which is why we
set `group_size` to 3 and sweep {2, 3, 4} rather than guessing. More directly relevant to inspection: among
the novel manifolds the paper recovers from DINO features are **shadows and lighting**. Illumination shift is
the canonical way a golden-reference differencing pipeline produces a false defect, so the confidence signal
we are attaching is one that has been shown to represent precisely the nuisance variable that breaks the
geometry stage.

**The SAE baseline, and why we expect to beat it.** The companion paper — Bhalla, Fel, Rager et al., *Do
Sparse Autoencoders Capture Concept Manifolds?*, arXiv:2604.28119 — shows SAEs recover continuous structure
suboptimally, fragmenting a manifold across atoms in a regime the authors term **dilution**, which is why
manifold structure is rarely legible in any single SAE feature. We do not take that on trust: the paper's
evaluation code is released at <https://github.com/goodfire-ai/sae-manifold> (MIT, pinned at
`f2632ddbb25f4c670254ceea999058b9ba4b0450`), and we reuse its minimal `BatchTopKSAE` as our SAE baseline and
its subspace-capture metric as the quantitative comparison. **Caveat we are explicit about:** that repository
targets Llama-3.1-8B with text-prompt manifolds, so what transfers is the SAE class and the metric definition,
not the data pipeline. We port the metric to our DINOv3 activations and say so in the report.

**BSF serves two independent purposes.** First, as a candidate confidence signal, where we expect but do not
assume better out-of-distribution generalisation than a raw activation probe. Second, and independently, as the
**explanation surface for human escalation**: the block coordinate lets the review UI show an operator *where
within* a learned concept an ambiguous region falls, which no other component in our stack can produce. The
second purpose stands even if the first does not, so we do not gate BSF's inclusion on AUROC alone.

**Risk control.** The linear probe is built first regardless — it costs roughly a day and guarantees the agent
loop has a working confidence signal from week 2, so that any difficulty with BSF cannot cascade into the
integration and deployment weeks. We report both probes side by side. If BSF does not beat the linear baseline
on OOD separation, we say so plainly and retain BSF for the explanation surface; a measured negative result is
reported as a result, not omitted.

---

## 5. Architecture: Local Baseline and Planned AWS Deployment

### 5.1 Development and evaluation baseline (on-premises, in use today)

We have no AWS account at this time, so the system is built and evaluated on hardware we own: a single
**NVIDIA DGX-1 node with 8x Tesla V100 32 GB**, with the pipeline pinned to **one GPU**
(`CUDA_VISIBLE_DEVICES=0`) so the remaining cards stay free for other tenants and the throughput claim holds
on a single-accelerator budget.

| Stage | Local runtime | AWS target (Section 5.2) |
|---|---|---|
| Intake | FastAPI service on the DGX host | API Gateway + Lambda |
| OpenCV 5 perception | Host Xeon cores, pinned core set | EC2 Graviton4 on the COOL AMI |
| Confidence sidecar | Frozen DINOv3 + probe + BSF on GPU 0 | EC2 x86 GPU instance |
| Planner | Deterministic policy engine on host CPU, optional local open-weights LLM for escalation rationale | Amazon Bedrock |
| Verdicts and traces | SQLite in WAL mode | DynamoDB |
| Frames and artifacts | Content-addressed local filesystem | S3 |
| Observability | JSONL event log + Prometheus/Grafana | CloudWatch |
| Human review queue | Static site served by the intake process | S3 + CloudFront |

**Volta constraints, handled up front.** V100 is compute capability SM 7.0: it has no BF16, and
FlashAttention-2 requires SM 8.0 or newer. We therefore run autocast in **fp16** rather than bf16 and use the
**PyTorch SDPA memory-efficient backend** rather than FlashAttention. The PyTorch/CUDA build is pinned to one
that ships `sm_70` kernels, and startup asserts on `torch.cuda.get_device_capability()` so a silent CPU
fallback cannot masquerade as a slow GPU.

**Consequence for the featured paths.** The Agentic Vision path — the primary one — runs end to end on this
hardware with no cloud dependency. **Best Use of COOL cannot**: COOL exists only for Graviton, so that path is
live only in the AWS deployment below and is therefore conditional on this grant.

### 5.2 Planned AWS architecture and services

```
                            +--------------------------------------+
   Operator / camera ------>|  API Gateway  ->  Lambda (intake)     |
   (upload or live feed)    +------------------+-------------------+
                                               |  frame + part class
                                               v
                         +-----------------------------------------+
                         |  EC2 Graviton4 - COOL AMI (OpenCV 5)     |   ARM PATH
                         |  calibrate -> homography align -> absdiff|
                         |  -> morphology -> contours -> metrology  |
                         |  -> cv::dnn defect classification        |
                         +------------------+----------------------+
                                            | aligned frame + verdict
                                            v
                         +-----------------------------------------+
                         |  EC2 (x86, GPU) - Confidence Sidecar     |   x86 PATH
                         |  frozen DINOv3 -> patch activations      |
                         |  -> linear probe / BSF                   |
                         |  -> confidence + OOD score + evidence    |
                         +------------------+----------------------+
                                            | confidence state
                                            v
                    +------------------------------------------------+
                    |  Agent planner - Amazon Bedrock                |
                    |  ACCEPT | RE-CAPTURE (loop back to ARM path)   |
                    |         | ESCALATE -> human approval gate      |
                    +-------+--------------------------+-------------+
                            |                          |
                            v                          v
                  Amazon DynamoDB              Human review queue
                  (verdicts + decision          (static site on S3 +
                   traces, replayable)           CloudFront; shows the
                            |                    probe's visual evidence)
                            v
                  Amazon CloudWatch - structured traces, latency,
                                      escalation-rate dashboards

   Amazon S3 - frames, golden references, defect masks, model artifacts
   IAM        - per-component least-privilege roles; no shared credentials
```

**Documented hybrid architecture.** The OpenCV 5 workload runs on AWS Graviton4 under COOL; backbone inference
runs on x86. This split is deliberate and is exactly the hybrid arrangement the Best Use of COOL rules permit,
with COOL executing the claimed core image workload on the Arm component.

**Services:** EC2 (Graviton4 + x86), Amazon Bedrock, AWS Lambda, API Gateway, Amazon S3, Amazon DynamoDB,
Amazon CloudWatch, AWS IAM, Amazon CloudFront.

**Cost discipline.** Baseline development costs the grant nothing: probe fitting, the full VisA evaluation
sweep, and the agent loop all run on our own DGX at zero marginal cost. On AWS we will not run an always-on
GPU endpoint either — the intake path and review queue stay warm on Lambda, S3, and CloudFront at negligible
cost, and the Graviton and x86 instances start on demand for benchmark runs and the judge demonstration. The
requested grant is budgeted primarily against **Graviton benchmarking hours** and **Bedrock planner calls**,
which is the work our own hardware cannot do.

---

## 6. Target Users and Beneficiaries

- **Primary — manufacturing QA engineers** at small and mid-sized producers who cannot fund a bespoke machine
  vision integration, and who need a system whose autonomy they can tune against a stated accuracy floor.
- **Secondary — line operators** receiving escalations. Their experience is the product's real UX: a short,
  ranked queue of genuinely ambiguous cases, each with the visual evidence that triggered it, rather than a
  firehose of low-confidence alerts.
- **Tertiary — quality and compliance auditors**, who gain a replayable trace for every autonomous verdict,
  including which frames the system declined to judge and why.
- **Beyond manufacturing** — the escalation architecture is domain-agnostic. We will document what transferring
  it to another inspection domain requires.

---

## 7. Evaluation Method and Judge Demonstration

Evaluation is designed into the build from week one, not appended at the end.

**Quantitative results we will report:**

1. **Detection quality** — precision, recall, F1, and pixel-level AUROC on VisA held-out splits, with and without
   the confidence gate engaged. Reported **per class and split by rigid versus deformable subset**, never as a
   single averaged number that would obscure where reference alignment does not apply.
2. **The escalation trade-off curve (headline result)** — fraction of frames auto-accepted versus accuracy within
   the auto-accepted set, swept across confidence thresholds. This is the number an operator actually buys.
3. **Distribution-shift robustness** — probes trained on synthetic and augmented defects, evaluated on real VisA
   anomalies; the vision analogue of the synthetic-to-real generalisation result that motivated this design.
4. **Ablation** — probe-gated escalation versus raw classifier softmax versus no re-look at all, isolating how
   much of the gain comes from the interpretability signal rather than from re-capturing.
5. **Agent task success** — how often a re-look actually corrects an initially wrong verdict, and how often the
   system escalates cases it should have handled (over-escalation cost).
6. **Cost comparison** — probe sidecar versus a VLM-as-judge baseline, per 1,000 frames, in dollars and latency.
7. **COOL performance** — latency, throughput, and cost per 1,000 frames on Graviton4 with COOL against a stock
   OpenCV 5 baseline, with instance types, COOL version, inputs, and method published for reproduction.
8. **Failure gallery** — defect classes the system handles poorly and cases where the probe was confidently
   wrong, presented plainly.

**Judge demonstration.** A judge-accessible web endpoint where a judge uploads or selects a frame — including
deliberately out-of-distribution frames we provide — and watches the full loop execute live: OpenCV 5 verdict,
confidence score, the agent's chosen action, the re-look, the changed verdict, and, where triggered, the
escalation with its evidence. The decision trace is visible in the UI. We will additionally offer a live
screen-share walkthrough. The repository ships pinned dependencies, infrastructure-as-code, a one-command
deploy, and a test suite; we will rehearse a clean clone-and-deploy from scratch before submitting.

**Licensing of everything we ship:** VisA (CC BY 4.0), DINOv3 (Meta DINOv3 License, access-gated),
Block-Sparse Featurizer
(<https://github.com/goodfire-ai/block-sparse-featurizer>, MIT), SAE baseline and subspace-capture metric
(<https://github.com/goodfire-ai/sae-manifold>, MIT),
OpenCV 5 (Apache 2.0), COOL via AWS Marketplace under its listing terms.

---

## 8. Featured Path Declaration

**Both paths.**

**Agentic Vision (primary).** OpenCV 5 output is the control input to the planner. We will submit the required
agent workflow diagram covering perception, decision, and action, plus recorded traces demonstrating that a
change in the OpenCV 5 result changes a subsequent tool call — together with our evaluation of task success,
failure handling, observability, and the human approval gate.

**Best Use of COOL (secondary).** COOL executes the core image workload on AWS Graviton4, the Arm component of
our documented hybrid architecture. We will submit the COOL version and instance configuration, a reproducible
benchmark method with inputs and baselines, and evidence that COOL is executing the claimed workload rather than
merely being installed.

---

## 9. Team Bio

> `[FILL — the grant is scored on team strength, so be specific: for each member give name, role on this
> project, relevant background in computer vision / ML / cloud engineering, and every prior hackathon or
> competition with placement where applicable. Name who owns the OpenCV pipeline, who owns the probe sidecar,
> who owns AWS deployment, and who owns the report and video.]`

**Team size: 2.** Every component has a named owner; nothing is unassigned.

**Aditya Kumar** (vasudeo118@gmail.com) — Perception and cloud infrastructure
Owns the OpenCV 5 pipeline (calibration, LightGlue alignment, differencing, segmentation, metrology), the local
DGX deployment and observability, and — if the grant lands — the Graviton4 / COOL port and benchmark.
`[FILL: background — relevant CV / systems / cloud experience, education or employment; prior hackathons and
competitions with results.]`

**Saumilya Gupta** (saumilya.ai@gmail.com) — Confidence sidecar, agent, and evaluation
Owns the frozen DINOv3 backbone and probe sidecar (linear probe and Block-Sparse Featurizer) on the local GPU,
the planner and escalation logic (local policy engine; Bedrock on the cloud port), the evaluation suite, and
the technical report.
`[FILL: background — relevant ML / interpretability / evaluation experience; prior hackathons and competitions
with results.]`

**Shared:** the human review UI, the five-minute video, and reproducibility (pinned dependencies, one-command
deploy, clean-clone rehearsal).

---

## 10. Build Schedule

| Week | Dates | Milestone |
|---|---|---|
| 1 | Sep 5-11 | VisA ingested; OpenCV 5 pipeline end-to-end locally; version compliance verified |
| 2 | Sep 12-18 | **Highest-risk work first** — activation extraction, linear probe, then BSF; AUROC on held-out and OOD splits, both probes reported side by side |
| 3 | Sep 19-25 | Probe wired into the agent loop; a re-look demonstrably changes a verdict. Grant check-in (window closes Oct 2) |
| 4 | Sep 26-Oct 2 | Local deployment hardened: policy-engine planner, SQLite trace store, escalation queue UI, replay tooling |
| 5 | Oct 3-9 | Full evaluation: baselines, shift table, ablation, escalation curve, failure gallery |
| 6 | Oct 10-16 | **If the grant landed:** Graviton4 + COOL port and benchmark, plus the storage-adapter swap to S3/DynamoDB/CloudWatch. **If not:** that week goes to the evaluation sweep and the failure gallery. Reproducibility pass either way — pinned deps, one-command deploy |
| 7 | Oct 17-23 | Technical report, architecture and agent-workflow diagrams, five-minute video |
| 8 | Oct 24-26 | Buffer; clean-clone deploy rehearsal; submit ahead of the deadline |

**Contingency.** If the schedule compresses, we cut physical re-capture and simulate re-looks as crop, zoom, and
re-parameterisation over stored high-resolution frames. The Agentic Vision qualifying requirement is that visual
evidence changes the next tool call — it does not require moving hardware. On the confidence signal we do not
compress: both probes are trained and reported, because the comparison between them is itself an evaluation
result the rubric asks for.

---

## Open Items Before Submitting This Proposal

- [ ] Register the team on Devpost
- [ ] Confirm with competition@opencv.org whether the grant proposal window is still open at this date
- [ ] Ask the organisers about the 50-team versus 55-grant discrepancy between the overview and prize list
- [ ] Add both member backgrounds (Section 9) — this section is what the grant is scored on
- [ ] Confirm DGX access: quota, GPU 0 availability, and a pinned PyTorch/CUDA build carrying `sm_70` kernels
- [ ] Confirm COOL Graviton4 AMI access and subscription terms on AWS Marketplace *(only if the grant lands)*
- [ ] Confirm AWS Free Tier credit eligibility for each member's account
