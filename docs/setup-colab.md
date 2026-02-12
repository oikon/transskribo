# Running Transskribo on Google Colab

## 1. Setup GPU Runtime

- Go to **Runtime > Change runtime type > T4 GPU**

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 2. Install System Dependencies

```bash
!apt-get update -qq && apt-get install -y ffmpeg
```

## 3. Mount Google Drive

Store audio files and output on Google Drive so they persist across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. Clone and Install

Use standard pip (`uv` has [known issues in Colab](https://github.com/astral-sh/uv/issues/4844)):

```bash
!git clone https://github.com/yourusername/transskribo.git
%cd transskribo
!pip install -e .
```

## 5. Set HuggingFace Token

Use Colab Secrets (key icon in the left sidebar) to add a secret named `HF_TOKEN`:

```python
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

## 6. Create Config

```python
%%writefile config.toml
input_dir = "/content/drive/MyDrive/audio_input"
output_dir = "/content/drive/MyDrive/transskribo_output"
model_size = "large-v3"
language = "pt"
compute_type = "float16"
batch_size = 8
device = "cuda"
log_level = "INFO"
max_duration_hours = 3
```

## 7. Run

```bash
!transskribo run --config config.toml
```

Generate a report after processing:

```bash
!transskribo report --config config.toml
```

## Gotchas

| Issue | Details |
|---|---|
| **Session timeout** | Free tier: ~90 min idle, 12 h max runtime. Process in batches. |
| **GPU memory** | T4 has 16 GB VRAM (more than RTX 4060). Default `batch_size=8` works fine. |
| **Python version** | Colab may run 3.10; Transskribo needs 3.12+. Check with `!python --version`. |
| **Resume after disconnect** | The hash registry tracks completed files. Re-run to skip already-processed files, or use `--retry-failed`. |
| **Model downloads** | First run downloads several GB of models. Cache is lost on session disconnect. |

## Getting Output

Since `output_dir` points to Google Drive, results persist across sessions. To download individual files:

```python
from google.colab import files
files.download('/content/drive/MyDrive/transskribo_output/your_file.json')
```

To zip and download everything:

```bash
!zip -r transskribo_results.zip /content/drive/MyDrive/transskribo_output
```

```python
files.download('transskribo_results.zip')
```
