"""Live collection layer for `strided watch`.

Pure-Python implementation of the architecture's "collector" role: poll metrics
endpoints (or replay captured files), parse into the canonical ``DiagnosisInput``,
and feed the existing engine on an interval. It deliberately depends only
*downward* — on ``schema``, ``parsers``, and ``engine`` — never on ``cli`` — so it
is the seam a future Rust collector can slot in behind without disturbing the
frontend.
"""
