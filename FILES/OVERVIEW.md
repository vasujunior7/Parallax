# Parallax — Solution Overview

*Decision summary for the OpenCV AI Competition 2026. The full submission document is `proposal.md`.*

> **The name.** Parallax is the apparent shift of an object when you view it from a different position — the
> reason two eyes see depth where one sees a flat picture. It is also, exactly, what this system does when it is
> unsure: change the viewpoint and look again. One view is not enough, and the system knows when it is looking
> at one.

---

## The decision, in one line

**Build for the Overall prize; take Agentic Vision as the primary special path and COOL as a secondary bolt-on.**

| | |
|---|---|
| Primary target | Overall Award ($5,000 / $3,000 / $2,000) — needs neither special path |
| Primary special path | **Agentic Vision** ($1,000) — structural, must be designed in from day one |
| Secondary special path | **Best Use of COOL** ($1,000) — mechanical, added in weeks 6-7 |
| Max stackable | $7,000 (one Overall + both specials) |

**Why Overall first:** $10,000 of the $12,000 in cash sits in the Overall awards, and the rules state plainly
that COOL and agentic methods are credited "only to the extent that they improve the project." A bolted-on agent
scores zero. So the agent has to be load-bearing or it should not exist.

**Why Agentic is the primary special:** its rubric puts 25% on orchestration and *appropriate autonomy* plus 15%
on failure handling, observability, and human control. That is 40% on the agent knowing when *not* to act — a
target almost no team will aim at deliberately.

**Why COOL is secondary, not skipped:** the qualifying bar is mechanical (run the core workload on Graviton,
benchmark against a baseline, document it). If the pipeline is already OpenCV-heavy it is roughly five days of
work for $1,000, and the same evidence feeds the 10% cloud-delivery line in the Overall rubric.

**Why COOL is now also grant-gated.** COOL runs only on Graviton, which runs only on AWS, and we have no AWS
account today. The build therefore targets hardware we own — a DGX-1 node with one V100 32 GB pinned — and the
entire Arm path is deferred to a cloud port that happens only if the compute grant lands. Nothing scoring on
the Overall or Agentic Vision rubrics depends on it. See `parallax_architecture.md` Sections 3 and 3b.

---

## What we are building

**Parallax** — an inspection service whose distinguishing feature is that it refuses to answer when it
should not. It runs on our own GPU today and is portable to the cloud; the refusal behaviour is the product
either way.

Every other entry will build something that detects defects. Nothing will detect **its own unreliability** —
which is the actual reason industrial vision systems fail to ship. A deployed system meets unfamiliar lighting,
an unexpected pose, a new supplier's finish, and returns a confident verdict anyway.

### The loop

```
  frame
    |
    v
  OpenCV 5 does the geometry
    align to golden reference -> difference -> segment -> measure
    |
    v
  verdict  +  frozen DINOv3 + probe: "is this frame like what I was validated on?"
    |
    v
  THAT SCORE selects the next action:

    confident + in-distribution  ->  ACCEPT, log the verdict, advance
    low confidence / OOD         ->  RE-LOOK: re-crop, new angle, new
                                     parameters -> re-run OpenCV -> re-score
    still uncertain after N      ->  STOP. Escalate to a human, with the
                                     visual evidence attached
```

**The re-look is the whole trick.** It satisfies the Agentic Vision qualifying test — the rules require that
visual evidence change what the system does next — and it is genuinely useful for the same reason a human
inspector tilts a part toward the light instead of guessing.

---

## Why this beats the modal entry

With roughly 1,164 registered participants, the median submission is predictable:

> YOLO or a Bedrock VLM + OpenCV for pre/post-processing + a chatbot that explains the detection.

That entry **fails the Agentic qualifying test outright** — the rules explicitly disqualify a chatbot that only
explains a fixed vision result. It also scores poorly on Technical Execution (30%, and the first tiebreak)
because OpenCV is plumbing around a model call.

Parallax inverts both:

- **OpenCV 5 does irreducible work.** Homography alignment, differencing, morphological segmentation, contour
  metrology. No learned component substitutes for it in our pipeline.
- **The "AI" is a cheap probe, not an expensive VLM.** This is the Goodfire/Rakuten pattern ported from language
  to vision, and it hands us the cost argument for free: their probes ran 10-500x cheaper than LLM-as-judge
  setups at comparable accuracy.

---

## The one number we are selling

Not accuracy. **The escalation trade-off curve:**

> *"At a 99% accuracy floor, this system handles 78% of frames autonomously and escalates 22%."*

That is the number a QA manager actually buys, it is what converts the automate-everything / review-everything
binary into a dial, and nobody else in the field will report it.

---

## The Goodfire component, and its fallback

The confidence signal comes from a frozen **DINOv3** backbone with a lightweight probe trained on its patch
activations. Both are in the build; the linear probe is simply built first:

1. **Linear probe** — the floor. Roughly one day. Near-certain to work. Sufficient to run the entire agent loop.
2. **Block-Sparse Featurizer** — the real signal. A BSF block reports both *how strongly* a concept is present
   and *where within that concept* the activation sits, so the escalation UI can say "this is the cracked end
   of the seam manifold" rather than just "low confidence." We use Goodfire's reference implementation,
   <https://github.com/goodfire-ai/block-sparse-featurizer> (MIT), pinned at commit `219f121e` — it ships the
   three featurizer variants and a trainer, but **no pretrained featurizers**, so training one is our work.
   Its `bsf.data` helper targets DINOv3 and we run that same backbone, so the shipped `pos_mean.npy` and the
   published results apply to us directly. See `parallax_architecture.md` 4C.

**The two papers this rests on, and the one result that matters most.**

- *Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds* (arXiv:2606.25234) — the
  BSF paper. Concepts come out **2-4 dimensional**, which fixes our `group_size`. And it recovers **shadow and
  lighting manifolds** from DINO features. That is the nuisance variable that breaks golden-reference
  differencing, so our confidence signal is aimed at the failure we actually have, not a generic one.
- *Do Sparse Autoencoders Capture Concept Manifolds?* (arXiv:2604.28119) — the SAE side. SAEs fragment
  manifolds across atoms (**dilution**), which is why no single SAE feature answers "is this lighting
  familiar?" Its code (<https://github.com/goodfire-ai/sae-manifold>, MIT) gives us the SAE baseline we
  measure against, so beating it is our result rather than a borrowed claim.

**BSF earns its place two ways, and only one of them is a bet.** As a confidence signal it may or may not beat
the linear probe on out-of-distribution separation — that is the open question. As the *explanation surface for
human escalation* it is unconditional: nothing else in the stack can tell an operator where within a learned
concept an ambiguous region falls. So we do not gate its inclusion on AUROC alone.

**We still build the linear probe first.** It costs about a day and guarantees the agent loop has a working
confidence signal from week 2, so a snag in BSF cannot cascade into the integration and deployment weeks. Both
probes are reported side by side; if BSF loses on OOD separation we say so and keep it for the explanations.
A measured negative result is still a result, and the rules require evaluation evidence including failure cases.

---

## Risks we have already priced in

| Risk | Mitigation |
|---|---|
| BSF loses on OOD separation | Linear probe built first guarantees a working signal by week 2; BSF stays for the explanation surface and the comparison is reported as a result |
| OpenCV 4.x shipped by accident | Alignment uses LightGlue + ALIKED/DISK, which exist only in OpenCV 5 — the dependency is functional, not nominal |
| Metric claims we cannot back | VisA has no intrinsics; we use a per-class scale factor and reserve full calibration for the live demo path, stated explicitly |
| Deformable parts break alignment | Scope align-and-difference to VisA's rigid classes; name the food classes as a stated limitation |
| Dataset licence conflict | VisA (CC BY 4.0), not MVTec AD (CC BY-NC-SA, conflicts with the licence granted to OpenCV/AWS) |
| No AWS account; credits may not land | The whole build runs on our own DGX-1 (one V100 32 GB pinned). AWS is a documented Phase-2 port, not a dependency — only the COOL benchmark is lost if credits never arrive |
| Shared DGX, other tenants on the box | Pipeline pinned to `CUDA_VISIBLE_DEVICES=0`; VRAM budget sized for a single 32 GB card, with the planner model evictable to host CPU |
| V100 is Volta (SM 7.0) | No BF16 and no FlashAttention-2 on this generation. fp16 autocast and the PyTorch SDPA memory-efficient backend are chosen up front; the `sm_70` build is pinned and asserted at startup |

---

## Timeline reality

The build phase opened **August 26**; final submissions are due **October 26, 11:59 p.m. Pacific**. That is
roughly seven weeks from today, not two months. The highest-risk work — activation extraction and the probe —
is scheduled for week 2, deliberately early, so a failure there leaves time to fall back.

Full week-by-week schedule is in `proposal.md`, Section 10.

---

## Before submitting

*Internal checklist — deliberately kept out of the grant application itself.*

- [ ] Add both member backgrounds to Section 9 of `GRANT_APPLICATION.md` — this is what the grant is scored on
- [ ] Register the team on Devpost
- [ ] Confirm with competition@opencv.org that the grant proposal window is still open at this date
- [ ] Ask the organisers about the 50-team versus 55-grant discrepancy between the overview and prize list
- [ ] Confirm DGX access: quota, GPU 0 availability, and a pinned PyTorch/CUDA build carrying `sm_70` kernels
- [ ] Confirm COOL Graviton4 AMI access and subscription terms on AWS Marketplace *(only if the grant lands)*
- [ ] Confirm AWS Free Tier credit eligibility for each member's account
- [ ] Verify both contact emails receive mail — winners must respond to notification or forfeit
- [ ] Submit via the JotForm linked from the competition page