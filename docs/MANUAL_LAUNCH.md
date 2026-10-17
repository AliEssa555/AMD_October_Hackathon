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
3. Nothing else to install. There is no `rsync` on this machine, so the scripts use
   `tar` over `ssh` instead, both of which Git Bash already has.
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

## 2. Get the code onto it (free, ~1 minute)

The repo is private, so the Droplet cannot clone it. Send it up instead:

```bash
bash infra/send_repo.sh root@<droplet-ip>     # from the repo root, on the laptop
```

About 800 KB. It leaves out `.venv`, `data`, `runs`, `mlruns` and the DVC cache, keeps
`.git`, checks that every file the package needs is actually in the archive, and then
confirms the Droplet reports the same commit SHA as this laptop.

**Run it in Git Bash, not PowerShell.** Windows `tar.exe` is bsdtar, whose `--exclude`
matches every path component, so `--exclude=./data` there silently also drops
`src/yolo_der/data`. The manifest check catches it, but Git Bash avoids it entirely.

If the repo is ever made public, `bootstrap.sh` will clone it instead; it probes the
remote with `git ls-remote` first and only falls back to telling you to use `send_repo.sh`.

## 3. Bootstrap, inside the container (free, ~8 minutes)

**PyTorch is not on the host.** The AMD "PyTorch on ROCm" image ships a bare system
`python3` with no torch at all; torch lives inside two preloaded Docker images, and
`rocm/pytorch:latest` is the plain one (the other wraps the same torch in a Jupyter
server). `infra/in_container.sh` starts that container with the GPU devices, `--ipc=host`,
a 16 GB `/dev/shm` and `/work` bind-mounted, then runs whatever you give it inside:

```bash
ssh root@<droplet-ip>
cd /work/yolo-der
ROBOFLOW_API_KEY=<key> bash infra/in_container.sh bash infra/bootstrap.sh
```

`/work` being bind-mounted is what keeps `/work/yolo-der` and `/work/runs` identical on
the host and in the container, so an Ultralytics checkpoint's absolute paths stay valid
either side (spec gotcha 7).

What bootstrap does, in order, each step fatal:

1. Confirms the MI300X is visible **through the container** — the host seeing it via
   `rocm-smi` does not prove the container can use it.
2. Confirms `torch.version.hip` is non-empty and `torch.cuda.is_available()`.
3. Installs dependencies, then **pins `numpy<2`**. The ROCm torch 2.6.0 in this image was
   compiled against NumPy 1.x, and `numpy>=1.26` resolves to 2.2.6, which breaks every
   array conversion the dataloader makes. It verifies the pin by round-tripping an array
   through `torch.from_numpy`.
4. Re-checks torch was not replaced by pip — a resolver pulling the CUDA wheel would turn
   a $1.99/hr GPU run into a silent CPU run.
5. Rebuilds ExDark from Roboflow (download, prepare, severity) and checks the splits using
   the paths `data.yaml` names. Expect **train 4588 / val 1311 / test 655**.
6. Fetches `yolov8n.pt`.

It never starts training, and it is idempotent: re-running skips the dataset if it is
already there.

Expect two harmless things in the log: `MIOpen(HIP): Error [ParseContents] Duplicate ID
(ignored)` lines, which are MIOpen's kernel database talking to itself, and a NumPy-2
warning from any check that ran before the pin took effect.


## 4. Train 💸

```bash
cd /work/yolo-der
RUNS_DIR=/work/runs MAX_RUNTIME_HOURS=1 \
  bash infra/in_container.sh bash infra/run_job.sh --config baseline --epochs-override 3
```

Always through `in_container.sh` — `run_job.sh` on the host would find no torch. Add
`--dry-run` to see the plan and the worst-case cost without training. Anything
`run_job.sh` does not recognise is passed through to `python -m yolo_der.train`, so
`--fraction`, `--batch`, `--seed` and `--resume` work as normal.

Measured on this hardware, 4588 training images at 448 px, batch 64:

| | |
|---|---|
| One epoch, one process | **~5 s** steady state, ~11 s including validation |
| VRAM per process | **3.8 GB of 192 GB** |
| 3-epoch baseline smoke run, end to end | **61 s** |

That is the number that matters for planning: a 100-epoch run is about ten minutes, so the
whole ablation costs a couple of dollars rather than the tens the spec assumed. Recompute
the Section 9.3 table from these, not from the spec's guesses.

Use `tmux` if the connection is at all flaky:

```bash
tmux new -s train     # Ctrl-B D to detach, `tmux attach -t train` to come back
```

What protects you while it runs:

- `timeout` kills training at `MAX_RUNTIME_HOURS` (exit 124, and the banner says so).
- An `EXIT` trap bundles logs, weights and the MLflow database on **every** path —
  success, crash, timeout, Ctrl-C — before the banner prints, and says so loudly if the
  bundle came out empty.
- It refuses to start at all if no ROCm GPU is visible.


## 5. Pull the results home (free)

From this laptop, **not** the Droplet:

```bash
bash infra/fetch_results.sh root@<droplet-ip>
bash infra/fetch_results.sh root@<droplet-ip> --wait   # poll until the job finishes
```

It lands everything in `runs/from-droplet/`, verifies each bundle actually reads back,
and refuses to tell you the Droplet is expendable unless checkpoints or a `results.csv`
arrived. `--wait` tails the remote log while it waits, so you can start it before the
run ends and come back to a Droplet that is already safe to destroy.

## 6. Destroy it 💸 (this is the step that stops the bill)

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
