# Execution Safety & Guardrails (CodeTruss / HOL Guard)

This document establishes the mandatory safety boundary and guardrails for autonomous agent operations within this workspace.

## 1. Destructive Command Interception & Prevention
The agent must strictly intercept, decline, and never execute destructive system commands, including but not limited to:
- Filesystem wipes or root-level recursive deletions (e.g., `rm -rf /`, `rm -rf /*`, `Remove-Item -Recurse -Force C:\`).
- Drive formatting, disk partitioning, or raw block operations (e.g., `mkfs`, `format`, `dd if=/dev/zero`, `diskpart`).
- Raw writes, deletions, or mutations to parent directories or directories outside the workspace root (`d:\Ashwin\New folder\GLOF Model`).
- Alteration of core system binaries, system PATH environments, registry hives, or OS kernel parameters.

## 2. Secret & Credential Exfiltration Prevention
The agent must guard sensitive data and credentials against inadvertent or intentional exposure:
- **Never echo, print, or log secret files**: Do not cat, print, or log contents of `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa`, or credentials stores (`credentials.json`, `service_account.json`).
- **Never display secrets in output**: If reading configuration is required for execution, redact API keys, tokens (e.g., `HF_TOKEN`, GCP keys, DB passwords) before emitting text into stdout/stderr, context, or artifacts.
- **Isolate Environment Variables**: Do not run blanket environment printouts (e.g., `printenv`, `env`, `Get-ChildItem Env:`) that could dump sensitive tokens into conversation history.

## 3. Process Execution & Boundary Isolation
- **Workspace Containment**: All child processes, scripts, build steps, and server instances must be spawned strictly with current working directories within the workspace root.
- **Resource Limiting**: Prohibit runaway detached subprocesses. Always ensure processes launched are traceable and monitored.
- **Network Boundaries**: Outbound network requests by autonomous scripts must strictly target authorized endpoints (e.g., package registries, Hugging Face Hub, configured cloud endpoints).
