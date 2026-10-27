# Running on the university OKD launcher

Every pod starts empty. Only `/mnt/*` survives, so everything (repo, Python, `docs.db`,
results) lives in `/mnt/primary/cdet`. A new pod just reconnects to it.

## Every time you start a pod

1. **Launch a pod** from the launcher. Ask for a GPU only if the run needs one.
2. **Open its terminal** (the launcher's `open_link`, or `oc rsh <pod-name>`).
3. **Paste this one line.** It pulls the latest code, installs anything missing, and loads the environment:

   ```bash
   bash /mnt/primary/cdet/hons-project/deploy/okd/setup.sh && . /mnt/primary/cdet/env.sh
   ```

4. **Run what you need** (see below).
5. **Collect results** (see below), then **stop the pod** so it is not holding a GPU.

The terminal is plain `sh`: load files with `. file`, not `source file`.

## First time ever (nothing on the volume yet)

`/mnt/primary/cdet/hons-project` does not exist yet, so fetch the script directly:

```bash
curl -fsSL https://raw.githubusercontent.com/davidob05/hons-project/main/deploy/okd/setup.sh | bash && . /mnt/primary/cdet/env.sh
```

This clones the repo, installs Python 3.13 (via `uv`) and Java if missing, installs the
package, downloads `eng-docs.jsonl` (3.4 GB) and builds `docs.db` (4.3 GB). It needs about
8 GB free (`df -h /mnt/primary`); if short, use another volume with
`DATA=/mnt/datasets/cdet` on every command. Run it once; later pods skip all of this.

If the repo is private (the `curl` gives a 404), clone with a token first, then run `setup.sh`:

```bash
mkdir -p /mnt/primary/cdet
git clone https://<GITHUB_TOKEN>@github.com/davidob05/hons-project.git /mnt/primary/cdet/hons-project
bash /mnt/primary/cdet/hons-project/deploy/okd/setup.sh && . /mnt/primary/cdet/env.sh
```

## Running things

```bash
cd /mnt/primary/cdet/hons-project
bash deploy/okd/run.sh                                   # smoke test: BM25 example client, 3 days
bash deploy/okd/run.sh -- python -m hons.run configs/x.toml    # your own pipeline (once it exists)
```

`run.sh` starts the server on `127.0.0.1:8000`, runs the command after `--`, copies the
logs next to the output, and stops the server.

For a long run, put it in the background so a dropped terminal does not kill it:

```bash
nohup bash deploy/okd/run.sh -- <your command> > $DATA/last.log 2>&1 &
tail -f $DATA/last.log          # watch it; Ctrl-C stops watching, not the run
```

## Results

Each run gets `/mnt/primary/cdet/runs/<timestamp>/`:

| File | What |
| --- | --- |
| `<runtag>.json` | the run file |
| `client.log` | what the client printed |
| `server.log` | server output |
| `<token>.log` | per-run server log |

```bash
ls /mnt/primary/cdet/runs/
tail -20 /mnt/primary/cdet/runs/*/client.log
```

Download to your laptop (needs `oc` logged in; the pod name changes every launch, find it with `oc get pods`):

```bash
mkdir -p ./runs
oc rsync <pod-name>:/mnt/primary/cdet/runs/ ./runs/
# fallback if the pod has no rsync:  oc cp <pod-name>:/mnt/primary/cdet/runs ./runs
```

## Updating the code

Push from your laptop. The next `setup.sh` does a `git pull` for you. To pull without the
rest of setup: `git -C /mnt/primary/cdet/hons-project pull`.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `source: not found` | the terminal is `sh`; use `. file` |
| `curl` gives 404 | repo is private; clone with a token (see above) |
| `/mnt/primary/cdet` missing in a new pod | the volume was not mounted; check the pod YAML lists `l426davidhomeclaim` |
| Setup warns `NOT persistent` | the data directory is not writable; check `df -h /mnt/*` |
| `oc` says Unauthorized | login token expired; copy a fresh login command from the web console |
| Pod disappeared mid-run | the launcher has idle/timeout limits (`timeout`, `idle_time` annotations); results already written to `/mnt/*` are safe |
| `server died:` from `run.sh` | read `runs/<timestamp>/server.log` |

## Layout on the volume

```
/mnt/primary/cdet/
  hons-project/   git clone (code)
  venv/           Python 3.13 environment
  tools/          uv, Python, Java
  db/             docs.db, eng-docs.jsonl, settings.toml, logs/ (the server's working dir)
  runs/           one folder per run
  env.sh          loaded with ". env.sh"
```
