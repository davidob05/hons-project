# Running on the university OKD launcher

Everything lives on the persistent volume `/mnt/primary/cdet`, so a fresh pod only needs
this one command (first time and every time after; it skips what is already done):

```bash
curl -fsSL https://raw.githubusercontent.com/davidob05/hons-project/main/deploy/okd/setup.sh | bash && . /mnt/primary/cdet/env.sh
```

Private repo, or no internet to raw.githubusercontent.com? Clone once, then use the local copy:

```bash
git clone https://<GITHUB_TOKEN>@github.com/davidob05/hons-project.git /mnt/primary/cdet/hons-project
bash /mnt/primary/cdet/hons-project/deploy/okd/setup.sh && . /mnt/primary/cdet/env.sh
```

On any later pod the local copy works too (and does a `git pull` first):

```bash
bash /mnt/primary/cdet/hons-project/deploy/okd/setup.sh && . /mnt/primary/cdet/env.sh
```

## Running things

```bash
cd /mnt/primary/cdet/hons-project
bash deploy/okd/run.sh                                  # smoke test: BM25 example client, 3 days
bash deploy/okd/run.sh -- python -m hons.run configs/x.toml
nohup bash deploy/okd/run.sh -- <cmd> > $DATA/last.log 2>&1 &   # survives a dropped terminal
```

Outputs: `/mnt/primary/cdet/runs/<timestamp>/`. The terminal is plain `sh`: use `. file`, not `source file`.
