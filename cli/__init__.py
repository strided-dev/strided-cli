"""strided CLI.

Thin orchestration: parse user inputs, fan them into one DiagnosisInput,
hand to the engine, format the report.
"""

# Lives here (not in main.py) so both the entry point and the renderer can
# import it without an import cycle.
CLI_VERSION = "0.1.0"

