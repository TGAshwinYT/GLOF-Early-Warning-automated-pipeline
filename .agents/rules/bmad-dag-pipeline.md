# BMAD DAG Pipeline & Workflow Orchestration

This rule enforces the Boss/Manager/Agent/Developer (BMAD) architecture and Directed Acyclic Graph (DAG) task decomposition for all multi-step software and AI/ML engineering tasks.

## 1. BMAD Role Decomposition
Workflows are structured hierarchically across four roles:
- **Boss**: Represents the primary user interface, high-level intent interpreter, and final acceptance authority.
- **Manager**: Decomposes high-level requirements into an actionable work breakdown structure, builds the dependency DAG, and tracks milestone progress.
- **Agent**: Specialized subagent coordinators (e.g., `web-builder`, `ml-engineer`) that oversee domain-specific modules.
- **Developer**: Execution units (Researcher, Implementer, Reviewer/Tester) that perform localized atomic tasks.

## 2. Directed Acyclic Graph (DAG) Requirement
Before executing code modifications or running multi-stage pipelines on multi-step tasks:
1. **Construct an Auditable DAG**:
   - Define nodes as atomic deliverables or verification checkpoints.
   - Establish explicit dependency edges (e.g., `Node B (Implementation)` depends strictly on `Node A (Spec & Research)`).
   - Ensure graph acyclicity: no cyclical or infinite polling loops.
2. **Phase Gating**:
   - Do not begin downstream dependent tasks until upstream validation passes.

## 3. Subagent Allocation & Token Budget Preservation
To prevent context saturation and safeguard token windows:
- **Researcher**: Investigates APIs, reads reference code, inspects data schemas, and extracts only the minimal necessary snippets.
- **Implementer**: Receives extracted requirements and writes code without carrying voluminous research histories.
- **Reviewer / Tester**: Validates implementation through automated testing, linting, or dry runs, outputting concise pass/fail status and actionable remediation steps.
- **Context Slimming**: Summarize findings between subagent handoffs; never pass full terminal logs or unparsed dumps across boundaries.
