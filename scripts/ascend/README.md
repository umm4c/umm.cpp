# Ascend 310P manual runners

The four model-facing scripts run the validated CANN build inside
`umm-cann:8.5.0-mvp`. They mount the source tree and model read-only, mount only
the selected output directory read-write, pass through the requested Ascend
devices, and enable `GGML_SCHED_STRICT_ACCEL=1` by default.

The BAGEL runner defaults to `build-cann-bagel`. The U1.5 runner defaults to
`build-cann-rebase`, which is the currently validated U1.5 binary. Do not point
U1.5 at `build-cann-bagel`: its newer strict support check rejects U1.5's offset
RoPE during graph reservation. `BUILD_DIR` remains configurable for testing a
new unified build after that regression is fixed. The validated U1.5 binary
exposes `text`, `image`, and `think-image`; BAGEL exposes all seven modes.

## Launch scripts

```bash
# BAGEL text and image
scripts/ascend/run-bagel.sh text 'Describe Ascend NPU.'
STEPS=8 scripts/ascend/run-bagel.sh image 'A red apple on a wooden table'

# Validated single-card Q8-understanding/F16-generation package
MODEL_DIR=models/BAGEL-7B-MoT-Q8U-F16G-UMM DEVICES=2 \
  UNDERSTANDING_BACKEND=CANN0 VISION_BACKEND=CANN0 \
  GENERATION_BACKEND='diffusion=CANN0,vae=CANN0' \
  GENERATION_MAX_VRAM=CANN0=40 \
  WIDTH=1024 HEIGHT=1024 STEPS=8 \
  scripts/ascend/run-bagel.sh image 'A red apple on a wooden table'

# BAGEL understanding/editing; the third positional argument is the input image
scripts/ascend/run-bagel.sh understand 'What is shown?' outputs/apple.png
scripts/ascend/run-bagel.sh edit 'Turn the apple green' outputs/apple.png

# SenseNova U1.5
scripts/ascend/run-u15.sh text '用一句话介绍华为昇腾。'
STEPS=8 scripts/ascend/run-u15.sh image 'A red apple on a wooden table'
```

Use `--help` on either script for the complete environment-variable list. The
defaults are deliberately small (256x256 and 8 steps). Recommended validated
layouts are:

| Model | Physical devices | Logical layout |
| --- | --- | --- |
| BAGEL Q8 understanding + F16 generation | `DEVICES=2` | LLM, diffusion, vision, and VAE on CANN0; validated at 1024x1024 with VAE tiling |
| BAGEL F16 understanding + F16 generation | `DEVICES=2,3,4` | LLM/VAE CANN0, diffusion CANN1+CANN2, vision CANN1 |
| U1.5 F32 generation | `DEVICES=2,3` | LLM CANN0, diffusion CANN0+CANN1 with 12/38 GiB budgets |

The order in `DEVICES` defines the logical CANN indexes inside the container.
For example, `DEVICES=4,5,6` maps physical devices 4/5/6 to CANN0/1/2.

For BAGEL, `run-bagel.sh` enables VAE tiling automatically when either output
dimension is at least 1024. Override it with `VAE_TILING=0|1`; the defaults are
`VAE_TILE_SIZE=64` latent pixels and `VAE_TILE_OVERLAP=0.5`. Tiling affects only
VAE encode/decode and does not unload either model branch.

## Performance scripts

```bash
# One warmup and three measured cold-process runs
scripts/ascend/bench-bagel.sh image 'A red apple on a wooden table'
scripts/ascend/bench-u15.sh image 'A red apple on a wooden table'

# A shorter manual run
WARMUP=0 REPEATS=1 STEPS=1 scripts/ascend/bench-bagel.sh image 'smoke test'

# Full-size U1.5 run
WARMUP=0 REPEATS=1 WIDTH=2048 HEIGHT=2048 STEPS=50 \
  scripts/ascend/bench-u15.sh image 'A red apple on a wooden table'
```

Each run starts a new `umm-cli` process, so `elapsed_seconds` includes container
startup, model loading, inference, decoding, and output writing. Results are
written below `benchmark-results/`:

- `results.csv`: status and end-to-end wall time;
- `summary.txt`: min/mean/max across successful measured runs;
- `configuration.txt`: effective environment overrides;
- `logs/`: complete stdout/stderr for every run;
- `engine-timings.txt`: sampling and generation timings extracted from logs;
- `npu/`: one-second `npu-smi info` samples;
- `npu-before.txt` and `npu-after.txt`: health snapshots;
- `outputs/`: generated PNG and JSON metadata.

Use the same prompt, seed, dimensions, steps, CFG, device order, and model package
when comparing commits. Run performance tests on otherwise idle devices.
