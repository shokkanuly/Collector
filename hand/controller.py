"""HandController: hand thread, latest-wins queue, 30 Hz cap, letter/mirror modes.

See ARCHITECTURE.md §4.4. The camera loop never blocks on serial I/O
(CLAUDE.md rule 2).

Implemented in ROADMAP stage 5.
"""
