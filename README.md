# SpatialSemanticPipeline

从在线视频生成空间语义数据的本地管线：下载视频、检测场景、切分视频、估计相机轨迹、判断轨迹质量、提取导航提示，并调用 VLM 生成语义标注。

## 运行流程

```text
视频 URL
  → 下载视频
  → 场景检测
  → 视频切片
  → 相机轨迹估计
  → 轨迹质量判断
  → 导航提示提取
  → 有提示/无提示两类语义打标
  → output/
```

## 1. 准备运行环境

要求：Linux、Conda、NVIDIA GPU、可用的 VLM API，以及符合许可证要求的 VGGT-Omega 权重。

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate pipeline
```

以后重新进入项目时，只需加载已有环境：

```bash
conda activate pipeline
```

该环境已经统一包含 Python、PyTorch、CUDA 运行时、FFmpeg、Node.js 和本项目依赖，不需要切换到 TransNetV2 或 VGGT-Omega 的其他环境。

## 2. 准备第三方工具和权重

本仓库不提供 TransNetV2、VGGT-Omega 的源码或模型权重。运行安装脚本，从官方仓库取得经过本项目验证的固定版本：

```bash
bash scripts/setup_third_party.sh
```

VGGT-Omega 权重需要先在 Hugging Face 模型页面申请访问权限。获批后登录并下载全部模型权重：

```bash
hf auth login
bash scripts/download_models.sh
```

安装完成后，以下路径应当存在：

```text
tools/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
tools/VGGT-Omega/checkpoints/VGGT-Omega-1B-512/model.pt
```

## 3. 配置 API

复制密钥模板：

```bash
cp .env.template .env
```

编辑 `.env`，填入真实密钥：

```dotenv
VLM_API_KEY=your-api-key
```

每次打开新终端后，在运行管线前加载密钥：

```bash
set -a
source .env
set +a
```

## 4. 修改重要配置

所有常用参数都集中在 `configs/config.yaml`。相对路径以仓库根目录为基准。

首次运行检查：

| 配置项 | 需要填写或确认的内容 |
| --- | --- |
| `base_url` | VLM 服务地址 |
| `model` | VLM 服务支持的模型名称 |
| `api_key_env` | API 密钥环境变量名，默认保持 `VLM_API_KEY` |
| `transnetv2_path` | TransNetV2 推理代码目录 |
| `vggt_tool_root` | VGGT-Omega 代码目录 |
| `vggt_checkpoint` | VGGT-Omega 权重路径 |
| `paths.sources_file` | 视频 URL 列表，默认 `input/sources.txt` |
| `paths.output_root` | 最终输出目录，默认 `output` |

还可调整：

| 配置项 | 用途 |
| --- | --- |
| `youtube_download_height`、`youtube_download_fps` | 下载视频规格 |
| `scene_detection_threshold`、`scene_min_sec` | 场景检测阈值和最短场景时长 |
| `chunk_duration_sec`、`full_chunk_duration_sec` | 切片和语义打标时长要求 |
| `vlm_workers` | 同一视频内的 VLM 请求并发数 |
| `vlm_retry_attempts`、`vlm_timeout_sec` | VLM 重试次数和超时 |
| `temperature`、`max_tokens` | VLM 生成参数 |
| `judge` | 轨迹质量判断的完整参数组 |
| `hint` | 导航提示提取的完整参数组 |

`judge` 与 `hint` 各自拥有独立的一套参数，均直接位于 `configs/config.yaml` 的同名配置段中。若只是按默认流程使用，建议先保留默认值。

## 5. 填写视频地址

编辑 `input/sources.txt`，每行填写一个视频 URL。允许空行和以 `#` 开头的注释：

```text
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
```

## 6. 运行管线

确认当前终端已经加载 Conda 环境和 API 密钥：

```bash
conda activate pipeline
set -a
source .env
set +a
```

处理一个新视频：

```bash
spatial-semantic-pipeline run --max-videos 1
```

处理最多 20 个新视频：

```bash
spatial-semantic-pipeline run --max-videos 20
```

长时间运行可使用日志监控脚本。该脚本使用同一个 `pipeline` 环境，并继承当前终端中已加载的 `VLM_API_KEY`：

```bash
MAX_VIDEOS=20 bash scripts/run_pipeline.sh
```

日志写入 `logs/`。在 tmux 中运行时，也要先在对应窗口中加载 `.env`。

## 输出

```text
output/
├── consolidation.json
├── judge_results.json
├── hint_results.json
├── info/
├── scenes/       # 场景信息和相机位姿 CSV
├── semantics/    # 语义标注文本
└── chunks/       # 与标注对应、未打包压缩的 MP4 视频
```

完整流程成功后，最终结果只保留在 `output/`。运行期间生成的 `cache/`、`metadata/` 和 `output/.staging/` 会被清理；如果任务失败或中断，它们会暂时保留，以便再次执行相同命令时继续运行。

## 常见问题

### 无法使用 GPU

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 找不到第三方工具或模型

检查 `configs/config.yaml` 中的 `transnetv2_path`、`vggt_tool_root` 和 `vggt_checkpoint`，并确认对应文件实际存在。

### VLM 请求失败

依次检查：

- 当前终端是否执行过 `source .env`；
- `VLM_API_KEY` 是否有效；
- `base_url` 和 `model` 是否与服务匹配；
- 服务是否支持当前配置的 `vlm_input_mode`；
- 当前机器能否访问该 API。

### 中断后是否需要重新开始

不需要。重新加载环境和 API 密钥，再执行原命令即可。管线会从最近的安全阶段继续；全部输出成功后，中间状态会自动删除。

## 测试

```bash
conda activate pipeline
python -m unittest discover -s tests -v
bash -n scripts/run_pipeline.sh scripts/setup_third_party.sh scripts/download_models.sh
```
