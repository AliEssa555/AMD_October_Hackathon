# Launching a GPU Droplet by hand

Everything you have to do yourself, in order, with the two commands that cost money
marked. Keep this open in a tab while a run is going.

Why by hand: the DigitalOcean console will not issue an API token without a payment
method on file, so there is no token, so there is no scripted launch and no reaper. See
`CLAUDE.md` → "Approved deviation from hard rule 0.2" for what replaces them.

**The one rule that costs real money: powering a Droplet off does not stop billing.**
Only *Destroy* does. A Droplet left powered off overnight bills the same $1.99/hr as one
that is training.

---

## Before you start

| | |
|---|---|
| Console | <https://devcloud.amd.com/gpus/new> |
| Droplet list | <https://cloud.digitalocean.com/droplets> |
| Repo | <https://github.com/AliEssa555/AMD_October_Hackathon> |
| Credit | $100, expires **19 Oct 2026** |
| Price | **$1.99** per GPU-hour (verified on the create form; the spec's $2.59 was wrong) |
| Budget | ~50 GPU-hours total |

You need, in this order:

1. Your SSH **private** key on this laptop, matching the public key the console shows
   (`test`, id 60164577). Check with `ssh-add -l` or just look in `~/.ssh/`.
2. `ROBOFLOW_API_KEY` to hand — the Droplet rebuilds the dataset itself.
3. `rsync` available in Git Bash, for the pull home. `rsync --version` to check.
4. The run plan and its cost, from me in chat. **I will not create a Droplet; you do it
   in the console.** I will not even ask you to without printing hours × $1.99 first.

---

## 1. Create the Droplet

On <https://devcloud.amd.com/gpus/new>:

| Field | Value |
|---|---|
| Region | **Atlanta DC1** (`atl1`) — RIC1 shows "GPUs currently unavailable" |
| Plan | **MI300X ×1**, 192 GB VRAM, 20 vCPU, 240 GB RAM (the ×8 plan is greyed out) |
| Image | **PyTorch 2.6.0 on ROCm 6.4.0** (id `194144122`) |
| Authentication | **SSH key** → `test`. *Not* a password |
| Hostname | `yolo-der-gpu-1` |
| Tags | `yolo-der`, `ephemeral-gpu` — both, every time (hard rule 0.5) |
| Project | `My AMD Home` |
| Backups / monitoring | off |

Then write the time down. Literally — a sticky note with the UTC time you clicked
Create. It is the only clock you have, and the bill is per hour of existence, not per
hour of training.

## 2. Get the code and the data onto it (free, ~10 minutes)

Two routes. **Route B is the one that works right now**, because nothing has been pushed
to `AliEssa555/AMD_October_Hackathon` yet — `curl` of the raw file returns 404.

### Route A — if the repo is pushed and public

```bash
ssh root@<droplet-ip>                    # accept the host key
curl -fsSL https://raw.githubusercontent.com/AliEssa555/AMD_October_Hackathon/main/infra/bootstrap.sh -o bootstrap.sh
ROBOFLOW_API_KEY=<key> bash bootstrap.sh
```

### Route B — send the repo up from this laptop

```bash
# from the repo root, on the laptop
rsync -ah --info=progress2 --exclude .venv --exclude data --exclude runs \
  ./ root@<droplet-ip>:/work/yolo-der/

ssh root@<droplet-ip>
ROBOFLOW_API_KEY=<key> bash /work/yolo-der/infra/bootstrap.sh
```

`bootstrap.sh` finds the tree already in place and skips the clone. The excludes matter:
`.venv` is Windows binaries that are useless there, and `data` is rebuilt on the Droplet
anyway, which is far quicker than your upload.

### What it does either way

Checks the GPU **first** and refuses to go on if `torch.version.hip` is empty or no
device is visible — a non-GPU Droplet is worth discovering in minute one, not minute
forty. Then installs dependencies, re-checks that pip did not quietly replace the ROCm
torch build, rebuilds the dataset from Roboflow, and fetches `yolov8n.pt`.

It prints the exact `run_job.sh` line to use next. It never starts training.


## 3. Train 💸

```bash
cd /work/yolo-der
RUNS_DIR=/work/runs MAX_RUNTIME_HOURS=1 \
  bash infra/run_job.sh --config baseline --epochs-override 3
```

`run_job.sh` prints the run plan, the watchdog, and the worst-case cost, then starts.
Add `--dry-run` to see that plan without training. Anything it does not recognise is
passed through to `python -m yolo_der.train`, so `--fraction`, `--batch` and `--resume`
work as normal.

Run it under `tmux` if the connection is at all flaky:

```bash
tmux new -s train     # then Ctrl-B D to detach, `tmux attach -t train` to come back
```

What protects you while it runs:

- `timeout` kills training at `MAX_RUNTIME_HOURS` (exit 124, and the banner says so).
- An `EXIT` trap bundles logs, weights and the MLflow database on **every** path —
  success, crash, timeout, Ctrl-C — before the banner prints, and tells you loudly if
  the bundle came out empty.
- The run refuses to start at all if no ROCm GPU is visible.

## 4. Pull the results home (free)

From this laptop, **not** the Droplet:

```bash
bash infra/fetch_results.sh root@<droplet-ip>
bash infra/fetch_results.sh root@<droplet-ip> --wait   # poll until the job finishes
```

It lands everything in `runs/from-droplet/`, verifies each bundle actually reads back,
and refuses to tell you the Droplet is expendable unless checkpoints or a `results.csv`
arrived. `--wait` tails the remote log while it waits, so you can start it before the
run ends and come back to a Droplet that is already safe to destroy.

## 5. Destroy it 💸 (this is the step that stops the bill)

<https://cloud.digitalocean.com/droplets> → the Droplet → **Destroy** → Destroy.

Then confirm the list is empty. If you see "Powered off", you are still paying.

Afterwards, locally:

```bash
mlflow ui --backend-store-uri sqlite:///runs/from-droplet/runs/mlflow.db
```

---

## If something goes wrong

**I lost the SSH connection mid-run.** The run is still going if you used `tmux`;
`ssh` back in and `tmux attach -t train`. Without `tmux` it died with the shell — but
the `EXIT` trap still ran, so the bundle exists. Pull it and check the log.

**The watchdog fired (exit 124).** Expected if the job was simply longer than the
budget. `last.pt` is in the bundle; resume on the next Droplet with
`--resume /work/runs/<name>/weights/last.pt`. The paths are identical across Droplets on
purpose, because Ultralytics writes absolute paths into the checkpoint.

**`fetch_results.sh` says the bundle is corrupt or empty.** Do not destroy the Droplet.
`scp -r root@<ip>:/work/runs ./rescue/` by hand and look at what is there.

**I am not sure whether a Droplet is still running.** Open the Droplet list. That page is
the truth; there is no reaper to catch one you forget, which is the whole cost of not
having a token.

**I need to stop everything right now.** Destroy the Droplet in the console. You lose
whatever has not been pulled, which is at most the current epoch; a `last.pt` is written
every epoch.
