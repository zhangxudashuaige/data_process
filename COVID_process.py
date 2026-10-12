from pathlib import Path
import shutil

import h5py
import pandas as pd
from anndata.io import read_elem, write_elem
from IPython.display import display

# ---------- 1. 文件路径 ----------
source = Path(r"E:\网站安装的软件\haniffa21.processed.h5ad")
output = source.with_name("haniffa21.standardized.h5ad")
temporary = source.with_name("haniffa21.standardized.partial.h5ad")

if not source.is_file():
    raise FileNotFoundError(f"找不到原始文件：{source}")

# 防止覆盖已有结果
if output.exists() or temporary.exists():
    raise FileExistsError(
        "结果文件或上次运行的临时文件已存在。\n"
        f"结果文件：{output}\n"
        f"临时文件：{temporary}\n"
        "请先检查已有文件，或修改输出文件名再运行。"
    )


# ---------- 2. 读取细胞信息 ----------
print("正在读取 obs……", flush=True)

with h5py.File(source, "r") as f:
    obs = read_elem(f["obs"])

required = [
    "full_clustering",
    "Status_on_day_collection_summary",
    "patient_id",
    "Days_from_onset",
    "Outcome",
]

missing = [column for column in required if column not in obs.columns]
if missing:
    raise ValueError(f"缺少必要的原始字段：{missing}")

original_obs = obs.copy(deep=True)

print(f"细胞数量：{len(obs):,}")
print("\n原始 Outcome 分布：")
print(obs["Outcome"].value_counts(dropna=False).to_string())


# ---------- 3. 统一字段 ----------
# 保留所有原始字段，新增或更新统一字段
obs["cell_type"] = obs["full_clustering"].copy()

# 使用 object 类型，允许添加 recover 类别并保留缺失值
obs["severity"] = (
    obs["Status_on_day_collection_summary"].astype(object)
)

# 忽略 Outcome 两端空格，按 Home 精确匹配
home_mask = (
    obs["Outcome"]
    .astype("string")
    .str.strip()
    .eq("Home")
    .fillna(False)
)

obs.loc[home_mask, "severity"] = "recover"

# patient_id 原样保留
# timepoint 转为数值，缺失值继续保留
raw_time = obs["Days_from_onset"]

# 清理空格，保留原始字段
time_text = raw_time.astype("string").str.strip()

# 已明确的非数值标签，统一作为时间缺失
missing_labels = {
    "healthy", "lps", "non_covid", "not_known", "nan", ""
}

missing_mask = (
    time_text.isna()
    | time_text.str.lower().isin(missing_labels)
)

numeric_time = pd.to_numeric(
    time_text.mask(missing_mask),
    errors="coerce",
)

# 如出现新的未知标签，继续停止并提示
invalid_time = ~missing_mask & numeric_time.isna()

if invalid_time.any():
    examples = time_text.loc[invalid_time].unique().tolist()
    raise ValueError(f"发现新的未知时间标签：{examples[:20]}")

obs["timepoint"] = numeric_time.astype("float64")

print("\n作为时间缺失处理的原始值：")
print(raw_time.loc[missing_mask].value_counts(dropna=False).to_string())
# 转为分类类型，保留实际出现的全部类别
obs["cell_type"] = pd.Categorical(obs["cell_type"])
obs["severity"] = pd.Categorical(obs["severity"])

expected = {
    "Healthy", "Mild", "Moderate",
    "Severe", "Critical", "recover",
}

actual = set(obs["severity"].dropna().tolist())
unexpected = actual - expected

if unexpected:
    print(
        "\n注意：发现目标类别之外的 severity 值，"
        "已原样保留，未擅自归类：",
        sorted(unexpected, key=str),
    )


# ---------- 4. 检查处理结果 ----------
assert obs.index.equals(original_obs.index)
assert len(obs) == len(original_obs)
assert obs.loc[home_mask, "severity"].eq("recover").all()

# 检查其余原始字段保持不变
updated_columns = {"cell_type", "severity", "timepoint"}

for column in original_obs.columns:
    if column not in updated_columns:
        pd.testing.assert_series_equal(
            obs[column],
            original_obs[column],
        )

print(f"\nOutcome = Home 的细胞数量：{int(home_mask.sum()):,}")

print("\n处理后的 severity 分布：")
print(obs["severity"].value_counts(dropna=False).to_string())

unified_columns = [
    "cell_type", "severity", "patient_id", "timepoint"
]

print("\n统一字段缺失数量：")
print(obs[unified_columns].isna().sum().to_string())

print("\n统一字段预览：")
display(obs[unified_columns].head(10))


# ---------- 5. 复制文件并写入新的 obs ----------
# 复制整个文件，保留表达矩阵及其他数据结构
print("\n正在复制原文件，请等待……", flush=True)
shutil.copy2(source, temporary)

print("正在写入统一后的 obs……", flush=True)

with h5py.File(temporary, "r+") as f:
    del f["obs"]
    write_elem(f, "obs", obs)
    f.flush()


# ---------- 6. 重新读取并验证 ----------
with h5py.File(temporary, "r") as f:
    saved_obs = read_elem(f["obs"])

pd.testing.assert_frame_equal(saved_obs, obs)

# 验证成功后才使用正式文件名
temporary.rename(output)

print("\n处理完成，重新读取验证通过！")
print(f"保存位置：{output}")
print("原文件已保留；没有删除细胞或基因。")