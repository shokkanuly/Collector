# Project Notes

This file was created by Claude to confirm write access to the Collector folder.

## Current status (Stage 1 complete)
- Dataset: 765 rows, balanced (A=11 sessions, B/C/D/E=10 each), C-duplication bug fixed.
- Models: Random Forest 96.44%, PyTorch NN 95.11% on a session-held-out test split.
- Tests: 4/4 passing; dry-run test now isolated in a temp dir (no longer deletes real models).

## Next
- Stage 2: unseen-conditions validation (new lighting / angles / backgrounds).
