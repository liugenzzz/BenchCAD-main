# Vision2Code 本地模型评估说明

本文档说明如何使用 `tools/eval_local_model.py` 加载本地训练好的
HuggingFace 多模态模型，在 BenchCAD Vision2Code 任务上进行评估。

脚本不会修改原有 `main.py`、`pipeline/runner.py` 或云端模型路由。它会：

1. 读取 Vision2Code 数据。
2. 使用本地模型根据 2x2 CAD 视图生成 CadQuery 代码。
3. 执行生成代码得到 STEP。
4. 使用项目原有 IoU 或 composite 评分逻辑计算分数。
5. 写出 `results.jsonl` 和每条样本的代码/STEP/渲染图。

## 1. 数据格式

脚本支持两种输入方式。

### 方式 A：已物化的 Vision2Code 数据目录

这是脚本实际评估时使用的目录结构：

```text
data_dir/
  records.jsonl
  codes/
    sample_id.py
  steps/
    sample_id.step
```

`records.jsonl` 每行是一条 JSON，例如：

```json
{"record_id":"washer_000001_s20260505","family":"washer","code_path":"codes/washer_000001_s20260505.py","step_path":"steps/washer_000001_s20260505.step"}
```

如果你已经有这种目录，直接通过 `--data-dir` 指向它即可。

### 方式 B：本地 parquet 原始分片目录

你当前的数据目录属于这种格式：

```text
E:\航天\benchCAD\benchCAD_data\code_gen\data
  code_gen-00000-of-00018.parquet
  code_gen-00001-of-00018.parquet
  ...
```

这种目录不能直接作为评估数据目录使用。需要通过 `--parquet-dir` 传入，
脚本会读取 parquet 中的 `stem`、`family`、`code` 字段，并在 `--data-dir`
下生成可评估数据；如果 parquet 中存在 `composite_png` 或 `view_*_png` 图片字段，也会
同步物化到 `images/` 并写入 `records.jsonl`：

```text
--data-dir/
  records.jsonl
  codes/
  steps/
  images/
```

生成 `steps/*.step` 时会执行每条 ground-truth CadQuery 代码，因此首次物化
完整数据会比较耗时。后续只要 `records.jsonl` 已存在，脚本会直接复用
`--data-dir`，不会重复转换，除非显式传入 `--rebuild-data`。

## 2. 推荐运行命令

CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 python ./tools/eval_local_model.py \
--model-path /app/llamafactory/trainspace/output/qwen3-vl-8b/full/sft/checkpoint-4596 \
--parquet-dir /app/BenchCAD-main/Vision2Code/code_gen/data \
--data-dir /app/BenchCAD-main/Vision2Code/data_local_codegen \
--out-dir results \
--materialize-limit 400 \
--limit 400 \
--dtype bfloat16 \
--device-map auto \
--trust-remote-code

##自定义jsonl数据的执行指令如下：
CUDA_VISIBLE_DEVICES=2,3,4,6,7 \
python tools/eval_local_model.py \
  --model-path /app/llamafactory/trainspace/model/Qwen3-VL-8B-Instruct \
  --model-name Qwen3-VL-8B-Instruct \
  --data-file /app/BenchCAD-main/Vision2Code/cad_eval_400.jsonl \
  --out-dir /app/BenchCAD-main/Vision2Code/qwen8b_results_base_400 \
  --limit 400 \
  --score iou \
  --max-new-tokens 16000 \
  --exec-timeout 300 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code \
  --resume
  
CUDA_VISIBLE_DEVICES=2,3,4,6,7 \
python tools/eval_local_model.py \
  --model-path /app/llamafactory/trainspace/output/qwen3-vl-8b/full/sft/checkpoint-3056 \
  --model-name checkpoint-3056 \
  --data-file /app/BenchCAD-main/Vision2Code/cad_eval_400.jsonl \
  --out-dir /app/BenchCAD-main/Vision2Code/checkpoint_3056_results_epoch_2 \
  --limit 400 \
  --score iou \
  --max-new-tokens 16000 \
  --exec-timeout 300 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code \
  --resume
 
CUDA_VISIBLE_DEVICES=4,6,7 python ./tools/eval_local_model.py \
--model-path /app/llamafactory/trainspace/model/Qwen3-VL-8B-Instruct \
--parquet-dir /app/BenchCAD-main/Vision2Code/code_gen/data \
--data-dir /app/BenchCAD-main/Vision2Code/data_local_codegen \
--out-dir /app/BenchCAD-main/Vision2Code/bench_qwen8b_results_base \
--materialize-limit 400 \
--limit 400 \
--dtype bfloat16 \
--device-map auto \
--trust-remote-code

### 断点续评

自定义 JSONL 的长时间评估建议传入 `--resume`。脚本每完成一条样本就将结果
原子写入 `--out-dir/results.jsonl`。任务被终止后，使用相同的
`--model-name`、`--out-dir` 和 `--score` 再次执行同一命令，脚本会跳过已有
结果，仅运行尚未完成的样本。

默认情况下，包括失败状态在内的已有记录也会跳过。若希望保留成功结果、但
重跑之前的 `model_fail`、`no_code`、`exec_fail` 或 `score_fail`，同时传入：

```bash
--resume --rerun-failed
```
  
CUDA_VISIBLE_DEVICES=2,3,4,6,7 python ./tools/eval_local_model.py \
--model-path /app/llamafactory/trainspace/output/qwen3-vl-8b/full/sft/checkpoint-3056 \
--parquet-dir /app/BenchCAD-main/Vision2Code/code_gen/data \
--data-dir /app/BenchCAD-main/Vision2Code/data_local_codegen \
--out-dir bench_data_results \
--materialize-limit 400 \
--limit 400 \
--dtype bfloat16 \
--device-map auto \
--trust-remote-code

### 使用本地 parquet 目录首次评估小样本

```powershell
cd E:\航天\benchCAD\BenchCAD-main\Vision2Code

python tools\eval_local_model.py `
  --model-path D:\models\your-trained-vlm `
  --parquet-dir E:\航天\benchCAD\benchCAD_data\code_gen\data `
  --data-dir data_local_codegen `
  --out-dir results_local_your_model `
  --materialize-limit 100 `
  --limit 10 `
  --dtype bfloat16 `
  --device-map auto `
  --trust-remote-code
```

含义：

- 先从 parquet 中转换前 100 条样本到 `data_local_codegen/`。
- 再从这 100 条中评估前 10 条。
- 结果写入 `results_local_your_model/`。

### 已经物化后再次评估

```powershell
python tools\eval_local_model.py `
  --model-path D:\models\your-trained-vlm `
  --data-dir data_local_codegen `
  --out-dir results_local_your_model `
  --limit 10 `
  --dtype bfloat16 `
  --device-map auto `
  --trust-remote-code
```

这次不需要再传 `--parquet-dir`，因为 `data_local_codegen/records.jsonl`
已经存在。

### 重建物化数据

```powershell
python tools\eval_local_model.py `
  --model-path D:\models\your-trained-vlm `
  --parquet-dir E:\航天\benchCAD\benchCAD_data\code_gen\data `
  --data-dir data_local_codegen `
  --rebuild-data `
  --materialize-limit 100 `
  --limit 10 `
  --dtype bfloat16 `
  --device-map auto `
  --trust-remote-code
```

传入 `--rebuild-data` 后，即使 `records.jsonl` 已存在，也会重新从 parquet
生成 `codes/`、`steps/`、`images/` 和 `records.jsonl`。

## 3. 参数详解

### 模型相关参数

`--model-path`

必填。本地训练好的 HuggingFace 模型目录，例如：

```text
D:\models\your-trained-vlm
```

目录中通常应包含 `config.json`、模型权重、tokenizer/processor 配置等文件。
脚本会依次尝试使用以下 Transformers 自动类加载模型：

- `AutoModelForImageTextToText`
- `AutoModelForVision2Seq`
- `AutoModelForCausalLM`

`--model-name`

可选。写入结果文件的模型名称。如果不传，默认为：

```text
local/<模型目录名>
```

例如 `--model-path D:\models\qwen-vl-cad` 会默认记录为
`local/qwen-vl-cad`。

`--trust-remote-code`

可选开关。传入后，`AutoProcessor.from_pretrained` 和模型加载会启用
`trust_remote_code=True`。如果你的模型是 Qwen-VL、InternVL 或其他需要自定义
模型代码的结构，通常需要打开这个参数。

`--dtype`

可选。模型加载精度。取值：

- `auto`：默认值，由 Transformers 自动决定。
- `float16`：半精度，常用于 CUDA 推理。
- `bfloat16`：BF16，常用于支持 BF16 的新 GPU。
- `float32`：单精度，占显存更多，但兼容性更高。

`--device`

可选。模型放置设备。默认 `auto`。常见取值：

- `auto`：有 CUDA 就用 `cuda`，否则用 `cpu`。
- `cpu`：强制 CPU。
- `cuda`：使用默认 GPU。
- `cuda:0`：使用第 0 张 GPU。
- `cuda:1`：使用第 1 张 GPU。

如果设置了 `--device-map`，模型会交给 Transformers 按 device map 放置，
此时 `--device` 主要作为 fallback。

`--device-map`

可选。传给 Transformers 的 `device_map`。大模型建议使用：

```text
--device-map auto
```

这样可以让 Transformers 自动把模型放到可用 GPU/CPU 上。

### 生成相关参数

`--max-new-tokens`

可选。每条样本最多生成多少新 token。默认使用项目的
`DEFAULT_MAX_TOKENS`，当前为 16000。Vision2Code 需要输出完整 CadQuery 程序，
如果模型经常截断，可以调大；如果显存或时间紧张，可以调小。

`--do-sample`

可选开关。默认不采样，使用确定性生成。传入后启用采样。

`--temperature`

可选。采样温度，默认 `0.2`。只有传入 `--do-sample` 时才会使用。
温度越高，输出越随机；评估通常建议保持默认的确定性生成，不传
`--do-sample`。

### 数据输入和物化参数

`--data-dir`

可选。Vision2Code 评估数据目录。默认是：

```text
Vision2Code/test_data
```

如果目录下已有 `records.jsonl`，脚本会直接读取该目录进行评估。
如果目录下没有 `records.jsonl`，则需要同时传入 `--parquet-dir`，让脚本先
物化数据。

`--parquet-dir`

可选。本地 BenchCAD `code_gen/data` parquet 分片目录，例如：

```text
E:\航天\benchCAD\benchCAD_data\code_gen\data
```

当 `--data-dir/records.jsonl` 不存在，或者传入 `--rebuild-data` 时，脚本会
从该目录读取 `*.parquet` 并生成可评估数据。

`--max-shards`

可选。只读取前 N 个 parquet 分片。例如：

```text
--max-shards 2
```

会只读取排序后的前 2 个 `*.parquet` 文件。适合先做小规模转换测试。

`--materialize-limit`

可选。只从 parquet 中物化前 N 条记录。例如：

```text
--materialize-limit 100
```

表示最多生成 100 条 ground-truth 代码、STEP 和 records 记录。

注意：这个参数控制“生成数据目录”的规模；它不是评估条数。评估条数由
`--limit` 控制。

`--materialize-exec-timeout`

可选。物化 ground-truth STEP 时，每条 CadQuery 代码允许执行的最长秒数。
默认使用项目 `DEFAULT_EXEC_TIMEOUT`，当前为 300 秒。

如果某条 ground-truth 代码超时或执行失败，该条样本会被跳过，不写入
`records.jsonl`。

`--rebuild-data`

可选开关。默认情况下，只要 `--data-dir/records.jsonl` 已存在，脚本会复用
已有数据目录，不重复从 parquet 转换。

传入 `--rebuild-data` 后，会重新读取 `--parquet-dir` 并覆盖生成
`records.jsonl`、`codes/`、`images/` 和缺失或需要重建的 `steps/`。

### 评估子集参数

`--records`

可选。只评估指定的 `record_id`。可以传多个：

```powershell
--records washer_000001_s20260505 hex_nut_000006_s20260505
```

`--limit`

可选。限制实际评估条数。如果没有 `--seed`，取筛选后的前 N 条。

例如：

```text
--limit 10
```

表示只评估 10 条。

`--seed`

可选。和 `--limit` 一起使用时，从筛选后的记录中随机抽样 N 条，且抽样可复现。

例如：

```text
--limit 10 --seed 42
```

表示固定随机抽 10 条。

### 评分和执行参数

`--score`

可选。评分方式，默认 `iou`。取值：

- `iou`：使用 voxel IoU，和 Vision2Code 默认指标一致。
- `composite`：使用 BenchCAD composite 分数，融合 IoU、关键操作、特征 F1、
  Chamfer、Hausdorff 等指标。

`--exec-timeout`

可选。执行模型生成的 CadQuery 代码时，每条样本允许的最长秒数。默认使用
项目 `DEFAULT_EXEC_TIMEOUT`，当前为 300 秒。

如果模型输出代码执行失败或超时，该条样本状态会记录为 `exec_fail`，得分为 0。

### 输出参数

`--out-dir`

可选。评估输出目录。默认是：

```text
Vision2Code/results_local
```

输出结构：

```text
out_dir/
  results.jsonl
  outputs/
    <safe_model_name>/
      record_id.py
      record_id.step
      record_id.png
```

`results.jsonl` 每条记录包含：

- `record_id`：样本 ID。
- `model`：模型名称。
- `status`：运行状态，例如 `ok`、`model_fail`、`no_code`、`exec_fail`、
  `score_fail`。
- `iou`：voxel IoU。
- `score`：主评分，取决于 `--score`。
- `score_type`：`iou` 或 `composite`。
- `lat_s`：模型生成耗时。
- `prompt_tokens`、`completion_tokens`、`total_tokens`：能解析时记录 token 数。
- `cost_usd`：本地模型固定为 `null`。
- `err`：失败信息。
- `code_path`：模型输出代码文件相对路径。
- `step_path`：模型输出 STEP 文件相对路径。
- `png_path`：模型输出 STEP 渲染图相对路径。

## 4. 常见建议

首次使用建议先小规模运行：

```powershell
--max-shards 1 --materialize-limit 20 --limit 2
```

确认 parquet 能读取、ground-truth STEP 能生成、本地模型能正常推理后，再逐步
扩大 `--materialize-limit` 和 `--limit`。

如果显存不足，优先尝试：

```powershell
--dtype float16 --device-map auto
```

或减少：

```powershell
--max-new-tokens 4096
```

如果模型输出不是标准 Python fenced block，脚本仍会调用项目原有
`extract_code` 尝试提取代码；如果无法提取，会记录为 `no_code`。
