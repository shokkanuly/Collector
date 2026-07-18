# Roadmap: Full-Alphabet Expansion (A–Z)

Scope decision: **24 static letters now** (A–Z minus J and Z, which are motion
gestures in ASL fingerspelling and structurally need a temporal model).
J + Z are Stage 5, deferred to the Milestone-2 sequence model.

Baseline at time of writing: 5 letters (A–E), 765 rows, RF 96.44% / NN 95.11%
on a session-held-out split. Repo clean at commit `09a67b2`.

**Progress:** Stages 0–3 complete. Full 24-letter dataset collected and
evaluated — NN 87.9% test accuracy (`632814a`). Remaining: Stage 4 (docs/ship,
in progress) and Stage 5 (J/Z, deferred). Optional: recollect `F` to fix the
F→B confusion before final ship.

---

## Stage 0 — Codebase prep (no new data)  ✅ DONE (`1828027`)

**Goal:** the pipeline trains and runs for an arbitrary letter set, with honest
evaluation, before any collection time is spent.

Steps:
1. **De-duplicate `FeedForwardNN`** → single definition in `utils/model.py`;
   `train.py`, `app_opencv.py`, `app_streamlit.py`, `test_pipeline.py` import it.
   Architecture stays byte-identical so the existing `best_model.pth` still loads.
2. **Generalize `data_collection.py`** label prompt: accept any single letter
   A–Z except J/Z (print why those are excluded). Keep auto-incrementing
   session IDs.
3. **Three-way split in `train.py`**: session-based train/val/test
   (~70/15/15, min 1 session each). Checkpoint selection on *val* loss;
   test set touched once at the end. Report both val and test accuracy.
4. **Fix `app_streamlit.py` fallback**: error out if `label_map.joblib` is
   missing instead of assuming A–E.
5. Scale confusion-matrix `figsize` with class count; remove unused
   `import mediapipe` from `app_opencv.py`.
6. Keep `test_pipeline.py` green; update its synthetic dry-run to also cover
   a >5-class case.

**Depends on:** nothing.
**Done when:** full test suite passes; a synthetic 24-class dry run trains;
the existing 5-letter `best_model.pth` still loads and runs in `app_opencv.py`.

---

## Stage 1 — Risk-first pilot: the fist cluster (M, N, S, T)  ✅ DONE (folded into full collection; cluster passed: M/N F1 1.00, A/S soft)

**Goal:** learn whether the hardest letters are viable *before* the full
collection marathon. A/E already exist; adding M, N, S, T completes the
closed-fist cluster that landmark models confuse most.

Steps:
1. Collect M, N, S, T — 10 sessions × 15 frames each (~70 s per letter),
   varying angle/distance/lighting between sessions.
2. Commit the dataset immediately after collection (data-safety rule).
3. Retrain (Stage-0 pipeline); inspect per-class F1 and the A/E/M/N/S/T
   block of the confusion matrix.

**Depends on:** Stage 0.
**Done when:** per-class F1 for all six fist letters is known and a written
go/adjust decision exists (adjust = e.g. more sessions for that cluster,
feature tweaks). Rough bar: cluster F1 ≥ 0.85 to proceed unchanged.

---

## Stage 2 — Full collection (remaining letters)  ✅ DONE (`b654aaa`, 24 letters × 10 sessions, validated clean)

**Goal:** complete the 24-letter static dataset.

Letters: F, G, H, I, K, L, O, P, Q, R, U, V, W, X, Y — 10 sessions × 15
frames each. ~25–30 min total camera time, done in batches.

Steps:
1. Collect in batches of ~5 letters; **commit the CSV after every batch**
   (lesson learned: never let unteachable trims/losses eat uncommitted data).
2. After each batch: sanity-check `value_counts()` + sessions-per-label —
   every letter 10 sessions / 150 rows, no C-style duplication.

**Depends on:** Stage 1 go decision.
**Done when:** CSV has 24 letters × 10 sessions × 15 frames (≈3,600 rows,
plus A's existing extra session), balance check clean, all committed.

---

## Stage 3 — Retrain + honest evaluation  ✅ DONE (`632814a`, NN 87.9% test)

**Goal:** a defensible results section for 24 classes.

Steps:
1. Retrain RF + NN on the full set with the three-way split.
2. Report: val-selected checkpoint's accuracy on the untouched test set,
   per-class precision/recall/F1, full confusion matrix.
3. Document expected confusion pairs (fist cluster; K/V; G/H; U/R) and the
   observed ones — honesty about failure modes is a scoring asset.
4. Commit models + metrics together with the dataset state they came from.

**Depends on:** Stage 2.
**Done when:** `train.py` output shows all 24 labels in both val and test
splits, and README-able numbers exist (test accuracy + per-class table).

---

## Stage 4 — Apps, docs, ship  🔄 IN PROGRESS (README updated; live app camera-test pending)

**Goal:** the live demo and public repo reflect the 24-letter reality.

Steps:
1. Verify both apps run with 24 classes (HUD text, TTS debounce still sane
   when many letters are near-threshold).
2. README: scope = 24 static letters; J/Z limitation stated plainly with the
   motion explanation; updated numbers.
3. Commit + push everything as a coherent state.

**Depends on:** Stage 3.
**Done when:** a stranger can clone the repo, read why it's 24 letters,
run the demo, and see numbers that match the committed models.

---

## Stage 5 — J + Z motion module (deferred; Milestone 2)

**Goal:** the two dynamic letters, via a temporal model — also the seed of
the SenseBridge sequence architecture.

Sketch (not planned in detail yet): ring buffer of the last ~30 frames of
landmarks → sequence classifier (small GRU/LSTM, or DTW template match as
baseline) → same `Result`/output path. New collection mode records short
clips instead of frames.

**Depends on:** Stages 0–4 shipped.
**Not started** until then — this is the scope-creep gate.

---

## Explicitly out of scope for this roadmap
- SenseBridge core/ refactor (shared capture/TTS extraction) — separate plan.
- Multi-signer dataset expansion — valuable, separately scheduled.
- Streamlit feature work beyond the fallback fix.
