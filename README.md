# HomeServer

Scripts and references for turning a Mac Pro (2013) into a research home server.
All automation files now live in the `scripts/` directory and HTML references in
`docs/`.

## Directory layout
- `scripts/` – provisioning and utility scripts
- `docs/` – additional documentation

## Remote access
Run `scripts/setup_remote_access.sh` to enable SSH, JupyterLab, and the simple
experiment API. The script will optionally reset the current user's password so
you can regain remote access if it was lost.

## Basic setup
Execute `scripts/mac_pro_server_setup.sh` for base package installation and
`setup_experiment_env.sh`, `setup_monitoring.sh`, or `optimize_performance.sh`
as needed.
