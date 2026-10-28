# Milestone write-ups

Status records written as the work progressed in October 2026. Each one
describes the code as it was at its milestone; the current design is in
[the architecture document](../architecture.md) and the summary of every
milestone in [the development log](../development.md).

| Post | Milestones | Figures from |
| --- | --- | --- |
| [Baseline](baseline-2026-10.md) | S0: instruments, roofline, loss budget | [`make_figures.py`](make_figures.py) |
| [Fewer, fatter tasks](s2-fused-tasks-2026-10.md) | S2: fused small phases, attention by chunk | [`make_figures.py`](make_figures.py) `--baseline` |
| [A weight ring that does not pay](s3-weight-prefetch-2026-10.md) | S3: weight prefetch across barriers (negative) | [`make_s3_figures.py`](make_s3_figures.py) |
| [Attention, balance, and two probes that said no](s2b-to-s4-2026-10.md) | S2b, S2c, S6a, S4 | |

The S6 showcase figures (`figures/s6-*.svg`), drawn for the final post,
come from [`make_s6_figures.py`](make_s6_figures.py), run by
[the showcase command](../../experiments/showcase/README.md); the two
architecture figures come from
[`make_architecture_figures.py`](make_architecture_figures.py).

The measurement run directories the scripts read (`runs/…`) are not part of
the repository.
