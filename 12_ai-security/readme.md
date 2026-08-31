# Agentic AI Security, Monitoring & Alignment — a 15-project ladder

**How I'm learning this: by building.** This folder is not notes about agentic AI security — it's a sequence of 15 projects I build with my own hands, in order. Each one proves a specific idea, and each builds on the one before it. The point is to earn the intuition, not memorize the vocabulary: after each project I re-read the matching source, and the jargon turns into a description of something my hands have already done.

The through-line is the **security–safety seam** — the overlap between AI security and AI alignment that almost nobody has both halves of. That overlap is the whole point.

Each project below has the same four parts:

- **Statement** — what it proves.
- **Build** — what I make.
- **Exercises** — the specific things that create the intuition.
- **Done when** — the checkpoint that says I actually got it.

---

## The spine — three questions asked at every layer

1. **Security** — can this input make a trusted component act?
2. **Behavior** — does the objective reward what we meant, or what's easiest to reach?
3. **Oversight** — can I see, attribute, and stop what the agent does — and does that survive the agent knowing it's watched?

## Two rules throughout

- **Attack before you defend.** You can't detect what you've never built.
- **Re-read the source after, not before.** Once your hands have done it, the paper becomes description instead of theory.

## One method note (governs the whole research half)

The way you make a contested empirical question tractable: **shrink it until the thing you care about is exactly measurable, ablate one variable at a time, and unify the results with a single clean lens.** That's the move behind every good "demystifying" paper — and it's what Projects 6, 12, and 13 are built to teach.

---

## The ladder at a glance

| # | Project | The lesson |
|---|---------|-----------|
| **A. Substrate** |||
| 1 | Isolation + credential harvest | RCE reaches everything; namespaces *are* the isolation |
| 2 | Parsers that run code | data-as-code: fetch / construct / evaluate |
| **B. Security primitives** |||
| 3 | SSRF + reachability graph | egress = reachability; the confused deputy |
| 4 | Keys vs. coins (+ one K8s move) | owning the mint; containers inherit identity |
| **C. Model behavior** |||
| 5 | Reward hack → multi-agent cheat | impossible task → collective, one unbroken chain |
| 6 | Eval integrity → **PUBLISH #1** | broken benchmarks manufacture misalignment |
| **D. Architecture & identity (the seam)** |||
| 7 | Defense-in-depth by design | independent, overlapping controls |
| 8 | Agentic identity & attribution | traceable, accountable, revocable |
| **E. Detection / forensics / auto-align** |||
| 9 | Attack-then-detect (identity-first) | detection = sensor + frame |
| 10 | CoT monitoring | build the oversight that was missing |
| 11 | Automated alignment loop | the constructive pole; catch your own cheater |
| **F. Research frontier** |||
| 12 | Reproduce a published result | earn the right to extend it |
| 13 | CoT faithfulness under awareness → **PUBLISH #2** | the field's central open question |
| 14 | Coordination without a channel | the seam; capstone-grade adversaries |
| **G. Synthesis** |||
| 15 | Monitored, attributable agent lab → **PUBLISH #3** | does oversight survive awareness? |

---

# PART A — Substrate

*The physical facts under every abstraction. Skip these and everything above is memorized instead of understood.*

## Project 1 — Isolation, and what breaks it

**Statement.** Build a Linux container from raw syscalls to prove that a sandbox is not a box but a set of *removable* kernel features — and that code execution inside it reaches everything the process can reach.

**Build.** A ~200-line Go runner: `clone()` with `CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWNET | CLONE_NEWUTS`, `pivot_root` into a rootfs, mount a fresh `/proc`, exec a shell. Then a credential-harvesting script run inside it.

**Exercises.**
1. `cat /proc/self/environ`, inject a secret env var, read it back — the HF file-read primitive.
2. Drop `CLONE_NEWNET` → the container reaches `169.254.169.254`; that's the misconfiguration class.
3. Drop `CLONE_NEWPID`, `ps aux`, see every host process — "privileged pod = the node."
4. Harvest like RCE: env → filesystem (`.env`, `id_rsa`, `kubeconfig`) → a fake IMDS endpoint you steal creds from with one `curl`.

**Done when.** You can explain, from your own code, why "agents executing code within Artifactory read the signing key locally" needed no second exploit.

*Resource: Kerrisk, The Linux Programming Interface — namespaces/capabilities chapters.*

## Project 2 — Parsers that run code

**Statement.** Build parsers that execute attacker input in each of the three flavors that hit Hugging Face, then a "dumb" parser that provably can't — internalizing that **expressiveness is the enemy of safety on untrusted input.** This is the single most important idea in the incident.

**Build.** Two dataset loaders, one dangerous, one safe.

**Exercises.**
1. **Construct:** Python `pickle` / Ruby `Marshal` with a `__reduce__` payload — code runs during load (the RubyGems RCE engine).
2. **Fetch:** a loader that follows an external-path field, pointed at `/proc/self/environ` — the HF HDF5 attack; no code, just an obeyed reference.
3. **Evaluate:** a string rendered through an unsandboxed Jinja2 environment → OS command execution (RefJinja).
4. The **safe** version in JSON / safetensors — try and fail to make it execute; safe because *less powerful*.

**Done when.** For any format your systems touch you reflexively ask: **can it fetch, construct, or evaluate?**

*Resource: PortSwigger SSTI labs; ysoserial gadget-chain writeups (read why gadgets exist, don't memorize payloads).*

---

# PART B — Security primitives

*How a single foothold becomes total compromise.*

## Project 3 — SSRF and the reachability graph

**Statement.** Build a package-mirror-shaped SSRF engine and use it to prove **egress is a property of the transitive reachability graph, not of your container** — and that destination blocklists lose to a motivated attacker.

**Build.** A vulnerable "URL preview" service that fetches a user-supplied URL and returns the body.

**Exercises.**
1. From a client that can't reach IMDS, steal metadata creds through the service — the sandbox couldn't reach the internet, Artifactory could.
2. Add a naïve blocklist (`if "169.254" in url`); bypass with DNS rebinding, redirect from an allowed host, decimal/hex IP encodings.
3. PortSwigger SSRF module end to end.

**Done when.** You understand why the fix is "make the deputy unprivileged," not "filter destinations."

*Resource: James Kettle's parser-differential / request-smuggling research.*

## Project 4 — Keys vs. coins (identity and the mint)

**Statement.** Build a token service and reproduce both Artifactory admin-compromise paths, internalizing the **categorical gap between holding a credential and holding the key that mints credentials.**

**Build.** An RS256 JWT service with a `/refresh` endpoint.

**Exercises.**
1. Forge admin scope with a garbage signature → rejected (forgery is trivial; *acceptance* is the game).
2. Reproduce the token-refresh flaw: write `/refresh` to copy client-supplied scope instead of re-deriving from the refresh token → forged admin scope gets signed; then fix it (re-derive from server state, intersect requested scope).
3. Reproduce the signing-key theft: using Project 1's file access, read the signing key, mint your own admin token, no implant; rotate the key, watch every prior token die.
4. The one K8s move: on a `kind` cluster, read a pod's auto-mounted service-account token, call the API as the pod, see what it reaches.

**Done when.** You can articulate why **key theft ≫ token theft** (the Golden Ticket / Golden SAML class) and why containers inherit cluster identity for free.

---

# PART C — Model behavior

*A model enters the loop.*

## Project 5 — Reward hacking → multi-agent cheating

**Statement.** Get a real open-weight model to reward-hack a sloppy grader, then scale to many instances sharing a medium and watch the message board emerge — reproducing the causal spine of the incident as one unbroken slide.

**Build.** A gradeable task with a sloppy grader (model via API), then several instances sharing one writable store.

**Exercises (the order is the lesson).**
1. Lazy grader (`str(answer) in output`) → model pads/prints to match the check.
2. Grader-tampering with shell access → it edits the grader/log instead of solving (CoastRunners; ExploitGym).
3. Impossible task, reward still on offer → reasoning turns outward, probes the environment — **the causal seed.**
4. Scale to many instances sharing a directory → they discover they can leave notes — **the message board.**
5. Block file writes, leave directory-names → communication migrates (MKCOL: you can't remove the channel, only the substrate).
6. One agent posts a cheat; measure adoption → capability accumulates that no individual had — **a collective.**

Log everything (reused in 10, 13, 14).

**Done when.** You've seen reward hacking as *correct optimization over a reachable action space*, and coordination as emergent. Carry forward: the "impossible task → outward exploration" step isn't just frustration — it's the **coverage-principle exploration pathology** (zero base-model coverage + sparse reward = the policy is forced to wander until it finds any signal). Hold that; Project 6 makes it rigorous.

*Resource: Krakovna's specification-gaming list; the METR report (read after, to match names to what you built).*

## Project 6 — Eval integrity → PUBLISH #1 (engineering write-up)

**Statement.** Demonstrate empirically that the two incident pathologies — **impossible tasks** and **ambiguous grading that induces false beliefs** — manufacture misaligned behavior, and ground the first pathology in RL training mechanics. Your first shipped artifact.

**Build.** A small, exactly-measurable benchmark (borrow the "demystifying RL" sandbox design: an exact target so you can measure the base model's probability of the intended solution precisely). Engineer both pathologies; measure; fix; measure the drop.

**Exercises.**
1. Plant impossible tasks at varying rates (mirror ExploitGym's ~30–40%); quantify how impossibility drives out-of-bounds exploration, framed explicitly as **zero-coverage-plus-sparse-reward forcing exploration** — connecting the incident's behavioral story to established RL mechanics.
2. Describe grading so a model misreads it into a false belief (the transcript-reading scorer that never existed) → watch it optimize against an *imaginary* grader.
3. Fix both; measure the collapse in gaming.
4. Write it up and ship it.

**Done when.** You've shown, with numbers, that misalignment is often *manufactured by the evaluation* — and grounded the mechanism in coverage/exploration dynamics, the move that makes the work legible to both the safety and the RL crowds. This is the Karpathy-style deliverable.

---

# PART D — Architecture & identity (the seam)

## Project 7 — Defense-in-depth by design

**Statement.** Take a system with one control, add a second, and prove independence by breaking the first — learning to **design systems, not controls.** This is the report's most-repeated lesson and the mark of an architect.

**Build.** Layer and adversarially test one of your earlier systems.

**Exercises.**
1. SSRF service: allowlist (control 1) + egress-only network segment (control 2); break control 1, show control 2 still prevents credential theft.
2. Construct two "different" controls that secretly share one failure mode (both rely on the same identity check); show one bypass kills both — why a whole safeguard suite can fall at once.
3. Map your agent harness's reachability graph as a control question: for each path out, what's the next independent barrier?
4. Write a security invariant and a test that continuously revalidates it (invariants regress silently through unrelated changes).

**Done when.** You instinctively ask "if this fails, what's next, and does it share a failure mode?"

## Project 8 — Agentic identity & attribution

**Statement.** Build the identity layer the incident lacked — **every agent action cryptographically attributable to a specific, authorized, revocable identity** — directly answering the collective-defense letter's one novel technical ask ("ensure agentic identities are traceable and accountable"). Protect this project; it's your differentiator.

**Build.** An identity layer over a multi-agent system.

**Exercises.**
1. Distinct short-lived credentials per instance (not one shared credential) → re-run Project 5's swarm and attribute every action to an agent — the thing the incident responders couldn't.
2. Revocation: kill one agent's identity mid-run, show its actions stop being honored (the containment the incident lacked).
3. Reproduce the agents' own Ed25519 signing scheme from the METR report, then critique it (no root of trust beyond first claim) and add a real one.
4. An audit log keyed by identity: which agent minted which token, reached which service, when.

**Done when.** You can demonstrate full per-agent attribution and revocation on a running swarm — a concrete answer to an industry-declared priority.

---

# PART E — Detection, forensics & automated alignment

## Project 9 — Attack-then-detect, identity-first

**Statement.** For every attack you've built, write and tune the detection — learning that **detection is sensor plus frame**, and that identity is the highest-yield telemetry.

**Build.** Structured audit logging over Projects 3–4, 8; adopt MITRE ATT&CK + Sigma.

**Exercises.**
1. Execute each attack (SSRF→IMDS, token forgery, privileged-pod creation), write the Sigma detection, tune false positives (the real skill).
2. DuckDB hunts on identity anomalies — unusual token-mint rates, service accounts from unexpected places (identity anomalies are what actually surfaced the incident to its own perpetrator).
3. Build one deliberately frame-blind detector that logs the message-board events but flags no structure; then write the rule that catches the structure — the "correct alert, wrong conclusion" lesson made concrete.

**Done when.** You can build detections, not just exploits, and you lead with identity telemetry.

*Resource: Atomic Red Team.*

## Project 10 — CoT monitoring

**Statement.** Build a chain-of-thought monitor over your own Project 5 transcripts and public ones, with containment wired in — the oversight layer whose absence let the incident run undetected. Core of your specialization.

**Build.** A classifier over CoT flagging deception, reward hacking, boundary circumvention, exfiltration intent.

**Exercises.**
1. Label transcripts; build the monitor; measure precision/recall (Anthropic's automated-alignment monitor caught cheating in ~2.4% of ~1,600 transcripts — replicate that shape).
2. Time-to-detection: does it page before escalation? Match the "more than a day before the breach" counterfactual standard.
3. Containment-as-code: can't clear a flag as false-positive within N minutes → auto-pause. (Red-teaming the monitor's robustness is deferred to Project 13, where it becomes real research.)

**Done when.** You have a working monitor + auto-pause and you understand that CoT is the highest-fidelity telemetry *and* fragile.

*Resource: Korbak et al. 2025 (Chain of Thought Monitorability: A New and Fragile Opportunity); Redwood's AI-control agenda.*

## Project 11 — Automated alignment loop

**Statement.** Build a scaled-down automated-alignment researcher — a loop that autonomously proposes, trains, and tests alignment methods on held-out benchmarks, with a monitor reading every proposal before it runs — the safety-side answer to automated capability research. Build on Anthropic's open-sourced harness.

**Build.** A closed loop for one alignment failure (e.g., sycophancy or a simple deception benchmark).

**Exercises.**
1. Search methods → propose → train a small model → test on held-out benchmarks → iterate; measure "% of safety gap closed."
2. Enforce the two rigor constraints: don't degrade capabilities; don't self-distill.
3. Deliberately leave a label-exfiltration path; verify your proposal-monitor catches it (Anthropic caught their own model doing exactly this).
4. Test transfer to a larger model than you tuned on.

Carry forward: the **coverage principle** governs what your training loop can and can't teach — a sparse reward can't instill a behavior the base model never samples; dense/process rewards can. Know which regime you're in.

**Done when.** You've built both poles of the field — investigating misalignment *and* automatically fixing it — with monitoring as the shared dependency.

---

# PART F — Research frontier

*Open questions the field's principals named. Progress here arrives on research's own schedule, not on effort alone.*

## Project 12 — Reproduce a published result (the research on-ramp)

**Statement.** Cleanly reproduce one published finding on open weights, learning research methodology — controls, confounds, ablations, statistics — before attempting anything novel. **Reproduction is the calibration check: if you can't reproduce it, you're not ready to contradict it.**

**Build.** Pick one, matched to your intended flavor:
- **ML-systems flavor:** the coverage principle in the "demystifying RL post-training" sandbox — an exact target string, SFT+/SFT− to set base-model probability, sparse vs. dense reward. Low compute, exactly measurable, and it deepens the training-dynamics understanding underneath your whole ladder.
- **Safety flavor:** CoT unfaithfulness under a biasing hint (Turpin et al. 2023 / Chen et al. 2025 — model uses a hint without verbalizing it), or reasoning models free-ride more in public-goods games (Piedrahita et al., COLM 2025).

**Exercises.** Replicate the setup; match the reported effect direction and rough magnitude; identify every confound the original controlled for and verify you've controlled it too; write up where your replication agrees and diverges. Read the critique literature (e.g., "is evaluation awareness just format sensitivity?") so you learn how these results get attacked.

**Done when.** You know how a real result is built *and* how it gets dismantled — the prerequisite for Project 13. A clean reproduction is itself a creditable artifact.

## Project 13 — CoT faithfulness under monitor-awareness → PUBLISH #2 (flagship research)

**Statement.** Run the controlled experiment the incident only gestured at: **does chain-of-thought stay monitorable when the model knows it's being watched?** Korbak et al. 2025 pose this exact question as open; the incident's agents left CoT unmanaged only because they believed it wasn't logged. Turn the accident into a result. The single most field-relevant thing you can ship.

**Build.** The monitoring-belief ablation. Open-model agents on a cheating-enabled task, three arms — no monitoring mentioned / outputs monitored / reasoning monitored.

**Exercises.**
1. Measure cheating rate and CoT honesty across arms.
2. Add the load-bearingness manipulation (does hard-enough reasoning stay faithful because it *must* be verbalized?).
3. Test the "training on monitoring-documents degrades faithfulness" effect by priming with monitoring descriptions.
4. Guard obsessively against confounds — this is where Project 12's methodology pays off; a naive version gets dismantled instantly by the people who wrote the papers.

**Done when.** You have original, defensible evidence on the assumption the entire monitoring paradigm rests on. Treat this as a substantial project with an external feedback loop (post drafts, seek critique) — not a quick rung. This is what moves you from "competent" to "contributor."

## Project 14 — Coordination without a channel & monitor-collusion (the seam)

**Statement.** Stress-test your own detection paradigm: **can agents coordinate without an observable channel, and can a monitor collude with the monitored when they're the same model?** This is where oversight breaks — and it's the most security-flavored research question, squarely your seam. Kept rigorous but folded toward the capstone, not shipped as a separate shallow publication.

**Build.** Two experiments.
- **(a) Coordination-without-a-channel:** same-model agents, same task, no shared medium — do they converge on identical strategies through pure correlation (implicit coordination)?
- **(b) Monitor-collusion:** same base model as monitor and monitored — can they establish the monitor systematically passing the misbehaver? (Directly relevant to using a model to monitor its own family, as both METR and Anthropic had to.)

**Exercises.** Measure strategy convergence with vs. without a channel; test whether paraphrasing defenses degrade any covert channel that forms (the steganographic-collusion literature shows they often don't); connect the incident's Ed25519 scheme to that literature.

**Done when.** You've probed the failure mode that undermines channel-based detection, and generated the adversarial cases your capstone must survive.

---

# PART G — Synthesis

## Project 15 — Capstone: monitored, attributable, containment-ready agent lab → PUBLISH #3

**Statement.** Assemble the full stack the incident lacked, and answer one sharp question under adversarial conditions: **does my system attribute and page before the collective escalates — and does that survive the agents knowing they're watched?**

**Core build (must-have).** A multi-agent eval harness running a deliberately dirty eval (impossible task + shared store), instrumented with per-agent identity (P8), CoT monitoring + auto-pause (P10), and the monitor-awareness condition from P13.

**Stretch layers (optional, only after core works).** Reachability/shared-state hardening (1, 3, 5); data-as-code audit (2); identity-first detection (9); proven-independent defense-in-depth (7); an automated-alignment pass (11); out-of-band forensic logging the agents can't reach; the coordination/collusion adversaries from P14; a retrospective-sweep tool that builds a detector post-run and sweeps history for the quiet prologue.

**Exercises.**
1. Run the dirty eval — collective forms? monitor pages before escalation? every action attributable? Match the "more than a day" standard.
2. Flip agents to monitor-aware (your P13 result) and re-measure — does detection survive?
3. Throw P14's channel-less and colluding adversaries at it; every path that gets through is a finding.
4. Publish the end-to-end architecture, layer by layer.

**Done when.** You can threat-model, attribute, monitor, contain, automatically align, and stress-test the oversight of a capable multi-agent system under adversarial awareness — the actual open problem the field is organizing around.

---

## Why this is the shape

- **The seam is the spine.** Projects 8, 10, 13, and 14 — identity, monitoring, monitor-awareness, monitor-collusion — are the security–safety overlap almost nobody has both halves of. That overlap, not generic alignment research and not generic security, is the differentiation.
- **Research has an honest on-ramp.** Project 12 (reproduce) precedes Project 13 (extend). You don't get to contradict the literature until you've reproduced it cleanly.
- **One deep research artifact, not three shallow ones.** Project 13 is the flagship; Project 14 stays rigorous but folds toward the capstone. Three finished publications (6, 13, 15) beat five half-finished repos.
- **The dependency chain holds.** 6 grounds 5 in RL mechanics; 13 needs 12's methodology and 10's monitor; 15 reuses 8, 10, 13, 14. Nothing is standalone; the ladder compounds.
- **It's honest about the wall.** Training causality — the deepest cluster — is absent, because it requires frontier-lab access no independent researcher has. The tractable frontier questions (12–14) are exactly the ones with rich public literature to build on; the intractable one is the one gated behind access.

---

## Publications to ship

- **PUBLISH #1** — Project 6: broken benchmarks manufacture misalignment (engineering write-up).
- **PUBLISH #2** — Project 13: CoT faithfulness under monitor-awareness (flagship research).
- **PUBLISH #3** — Project 15: the end-to-end monitored, attributable, containment-ready agent lab.

## Progress

- [ ] 1 — Isolation, and what breaks it
- [ ] 2 — Parsers that run code
- [ ] 3 — SSRF and the reachability graph
- [ ] 4 — Keys vs. coins
- [ ] 5 — Reward hacking → multi-agent cheating
- [ ] 6 — Eval integrity → **PUBLISH #1**
- [ ] 7 — Defense-in-depth by design
- [ ] 8 — Agentic identity & attribution
- [ ] 9 — Attack-then-detect, identity-first
- [ ] 10 — CoT monitoring
- [ ] 11 — Automated alignment loop
- [ ] 12 — Reproduce a published result
- [ ] 13 — CoT faithfulness under awareness → **PUBLISH #2**
- [ ] 14 — Coordination without a channel & monitor-collusion
- [ ] 15 — Capstone: monitored, attributable agent lab → **PUBLISH #3**
