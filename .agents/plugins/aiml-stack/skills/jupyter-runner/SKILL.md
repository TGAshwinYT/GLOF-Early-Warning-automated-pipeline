---
name: jupyter-runner
description: Execution protocols for Jupyter (.ipynb) notebooks cell-by-cell without stripping cell metadata or execution histories.
---

# Jupyter Notebook Runner & Execution Protocol

This skill specifies safe, non-destructive execution protocols for interacting with and modifying `.ipynb` notebook files.

## 1. Non-Destructive Cell-by-Cell Execution
When modifying, adding, or evaluating cells in Jupyter notebooks:
1. **Preserve Metadata**:
   - Never wipe top-level notebook metadata (e.g., `kernelspec`, `language_info`).
   - Retain cell-level metadata (e.g., `cell_id`, tags, collapsed state, execution counts when non-conflicting).
2. **Preserve Execution History & Outputs**:
   - Do not perform blanket output clears (`nbformat` or tool stripping) unless the user explicitly commands a full notebook wipe.
   - Maintain historical outputs in undisturbed cells to preserve visual graphs, training loss curves, and diagnostic logs.
3. **AST / JSON Structural Manipulation**:
   - Treat `.ipynb` strictly as structured JSON adhering to the `nbformat` specification (v4).
   - Use dedicated Python tools (such as `nbformat`, `papermill`, or `ipykernel`) or precise JSON manipulations rather than blind string regex/token replacements that break notebook parsing.

## 2. Cell Execution Workflow
1. **Dependency Verification**:
   - Ensure the associated virtual environment kernel is active and registered before triggering executions.
2. **Sequential Execution**:
   - Execute cells strictly in top-to-bottom sequence unless evaluating an isolated idempotent utility cell.
3. **Execution Verification**:
   - Check cell execution return codes and error tracebacks. Halt immediately if a preceding cell raises an unhandled exception (`NameError`, `ImportError`, `CUDA out of memory`).
4. **Clean Serialization**:
   - Format JSON with 1-space indentation (`indent=1` as standard in modern Jupyter tools) and POSIX or CRLF line endings consistent with the repository.
