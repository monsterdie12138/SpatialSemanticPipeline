# SpatialSemanticPipeline

SpatialSemanticPipeline is a local pipeline that turns online videos into spatial and semantic annotations. It downloads videos, detects and clips scenes, estimates camera trajectories, checks trajectory quality, extracts navigation hints, and calls a vision-language model (VLM) to generate annotations.

The associated dataset is hosted in a public Hugging Face repository. [SpatialSemanticDataset](https://huggingface.co/datasets/MonsterDie/SpatialSemanticDataset)

## Workflow

```text
Video URL
  → Download video
  → Detect scenes
  → Create video clips
  → Estimate camera trajectories
  → Assess trajectory quality
  → Extract navigation hints
  → Generate semantic annotations with and without hints
  → output/
```

## 1. Set up the environment

You need Linux, Conda, an NVIDIA GPU, access to a VLM API, and VGGT-Omega weights that you can use under their license.

From the repository root, run the following commands.

```bash
conda env create -f environment.yml
conda activate pipeline
```

In subsequent sessions, activate the existing environment.

```bash
conda activate pipeline
```

The environment includes Python, PyTorch, the CUDA runtime, FFmpeg, Node.js, and the project dependencies. It also supports TransNetV2 and VGGT-Omega.

## 2. Install third-party tools and model weights

The setup script fetches the pinned versions of TransNetV2 and VGGT-Omega that this project has verified.

```bash
bash scripts/setup_third_party.sh
```

Request access to the VGGT-Omega weights on the Hugging Face model page. Once access is granted, sign in and run the model download script.

```bash
hf auth login
bash scripts/download_models.sh
```

After setup, check that these files exist.

```text
tools/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
tools/VGGT-Omega/checkpoints/VGGT-Omega-1B-512/model.pt
```

## 3. Configure API access

Copy the environment template.

```bash
cp .env.template .env
```

Set your API key in `.env`.

```dotenv
VLM_API_KEY=your-api-key
```

In each new terminal, export the variables before running the pipeline.

```bash
set -a
source .env
set +a
```

## 4. Review the configuration

Common settings are in `configs/config.yaml`. Relative paths are resolved from the repository root.

Review these settings before the first run.

| Setting | What to configure |
| --- | --- |
| `base_url` | VLM API base URL |
| `model` | Model name supported by the VLM service |
| `api_key_env` | Name of the API key environment variable. The default is `VLM_API_KEY`. |
| `transnetv2_path` | TransNetV2 inference code directory |
| `vggt_tool_root` | VGGT-Omega code directory |
| `vggt_checkpoint` | Path to the VGGT-Omega weights |
| `paths.sources_file` | Video URL list. The default is `input/sources.txt`. |
| `paths.output_root` | Final output directory. The default is `output`. |

Other settings control video downloads, scene detection, and annotation.

| Setting | Purpose |
| --- | --- |
| `youtube_download_height`, `youtube_download_fps` | Video download resolution and frame rate |
| `scene_detection_threshold`, `scene_min_sec` | Scene detection threshold and minimum scene length |
| `chunk_duration_sec`, `full_chunk_duration_sec` | Duration settings for clipping and semantic annotation |
| `vlm_workers` | Number of concurrent VLM requests for one video |
| `vlm_retry_attempts`, `vlm_timeout_sec` | VLM request retry count and timeout |
| `temperature`, `max_tokens` | VLM generation settings |
| `judge` | Trajectory quality assessment settings |
| `hint` | Navigation hint extraction settings |

The `judge` and `hint` sections have independent settings. Keep their defaults for a standard run.

## 5. Add video URLs

Edit `input/sources.txt` and enter one video URL per line. Blank lines and comments beginning with `#` are allowed.

```text
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
```

## 6. Run the pipeline

Activate the Conda environment and load the API key in the current terminal.

```bash
conda activate pipeline
set -a
source .env
set +a
```

To process one new video, run the following command.

```bash
spatial-semantic-pipeline run --max-videos 1
```

To process up to 20 new videos, run the following command.

```bash
spatial-semantic-pipeline run --max-videos 20
```

For longer runs, use the logging wrapper. It uses the same `pipeline` environment and inherits the current terminal's `VLM_API_KEY`.

```bash
MAX_VIDEOS=20 bash scripts/run_pipeline.sh
```

Logs are written to `logs/`. When using tmux, load `.env` in the pane where you run the pipeline.

## Outputs

```text
output/
├── consolidation.json
├── judge_results.json
├── hint_results.json
├── info/
├── scenes/       # Scene information and camera pose CSVs
├── semantics/    # Semantic annotation text
└── chunks/       # Individual MP4 clips corresponding to the annotations
```

After a successful run, the final results are in `output/`, and the pipeline cleans up `cache/`, `metadata/`, and `output/.staging/`. If a run fails or is interrupted, these directories are kept temporarily so the same command can resume the run.

## Troubleshooting

### GPU unavailable

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### Third-party tools or model weights are missing

Check `transnetv2_path`, `vggt_tool_root`, and `vggt_checkpoint` in `configs/config.yaml`, then verify that the referenced files and directories exist.

### VLM requests fail

- Check that you ran `source .env` in the current terminal and exported the variables.
- Check that `VLM_API_KEY` is valid.
- Check that `base_url` and `model` match your VLM service.
- Check that the service supports the configured `vlm_input_mode`.
- Check that the current machine can reach the API.

### Resume an interrupted run

Reactivate the environment, reload the API key, and rerun the original command. The pipeline resumes from the latest safe stage and removes intermediate state once all outputs are complete.

## Tests

```bash
conda activate pipeline
python -m unittest discover -s tests -v
bash -n scripts/run_pipeline.sh scripts/setup_third_party.sh scripts/download_models.sh
```