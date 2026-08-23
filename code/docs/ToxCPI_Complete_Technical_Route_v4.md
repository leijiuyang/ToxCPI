# ToxCPI：面向毒理学的化学品-蛋白质相互作用预测与结局关联框架

## 完整技术方案 v4.0

---

## 修订说明（v4.0 更新内容）

本版本根据模型迭代实验（E1-E6）和效应任务策略讨论进行了重大更新：

1. **效应预测任务重构**：放弃层次化预测（无法构造可靠的无效应负样本），改为两个独立二分类任务（Activation / Inhibition），与 DTIAM 保持一致
2. **化学品编码器更新**：放弃 ChemBERTa + AttentiveFP + ToxAlert 三视图方案（ToxPrint 导致 shortcut learning），采用 BerMol token 级预训练特征
3. **可解释性方案落地**：BerMol 的 Morgan radius=1 分词机制保证 token=原子一一对应，结合 RDKit 官能团池化实现毒理学可解释性
4. **模型架构对齐实际实现**：更新为经过验证的 E6 架构（BerMol tokens + ESM-2 residues + BilinearAttention）
5. **性能数据更新**：四场景全面超越 tipFormer 基线

---

## 一、研究定位与框架概述

### 1.1 研究定位

```
领域定位：计算毒理学 × 深度学习
目标期刊：Environmental Science & Technology (一区Top)
核心问题：预测环境化学品对人类蛋白质的效应类型（结合/激活/抑制），
         并通过多层次分析关联毒理学结局
```

### 1.2 与现有工作的关键差异

| 维度 | DTIAM (Nat Commun 2025) | tipFormer (ES&T 2025) | **ToxCPI (本研究)** |
|------|------------------------|----------------------|---------------------|
| **预测任务** | Binding + MoA（独立二分类） | 仅 Binding | **Binding + Activation + Inhibition** |
| **化学品表征** | 自研预训练 + CLS 全局向量 | BerMol CLS 全局向量 | **BerMol token 级（原子级）** |
| **蛋白质表征** | 自研预训练 + CLS 全局向量 | ESM-2 CLS 全局向量 | **ESM-2 残基级** |
| **交互建模** | 无（特征拼接 + AutoML） | CrossAttention（seq_len=1，伪序列级） | **多头 BilinearAttention（原子-残基级）** |
| **可解释性** | 无（黑盒 AutoGluon） | 无（全局向量交互） | **原子-残基注意力矩阵 + 官能团池化可视化** |
| **预测方法** | AutoGluon（stacking + bagging） | MLP 分类器 | **端到端深度学习** |
| **效应数据** | TTD 单一来源 | 不预测效应 | **7 个数据源整合** |
| **下游分析** | 无 | 通路富集验证 | **通路 + PPI + AOP 三维度** |

### 1.3 框架整体结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ToxCPI：面向毒理学的 CPI 预测与结局关联框架 (v4.0)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    Part 1: CPI 预测模型（核心）                        │ │
│  │                                                                        │ │
│  │   输入：化学品 SMILES + 蛋白质序列                                    │ │
│  │                    ↓                                                   │ │
│  │   模块 A：预训练特征编码                                              │ │
│  │   ├── 化学品：BerMol token 级（每个 token = 一个原子）                │ │
│  │   └── 蛋白质：ESM-2 残基级（每个位置 = 一个氨基酸）                  │ │
│  │                    ↓                                                   │ │
│  │   模块 B：多头 BilinearAttention 交互建模                             │ │
│  │   （原子-残基级交互，产出可解释注意力矩阵）                           │ │
│  │                    ↓                                                   │ │
│  │   模块 C：任务预测头                                                  │ │
│  │   ├── Binding 任务：二分类（是否结合）                                │ │
│  │   ├── Activation 任务：二分类（是否激活）                             │ │
│  │   └── Inhibition 任务：二分类（是否抑制）                             │ │
│  │                    ↓                                                   │ │
│  │   输出：P(binding), P(activation), P(inhibition)                      │ │
│  │         + 原子-残基注意力矩阵 A[i,j]                                  │ │
│  │         + 官能团-残基可解释性热图（RDKit 后处理）                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ↓                                        │
│                         预测的靶蛋白集合 + 效应类型                         │
│                                    │                                        │
│            ┌───────────────────────┼───────────────────────┐               │
│            ↓                       ↓                       ↓               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Part 2A:       │    │  Part 2B:       │    │  Part 2C:       │        │
│  │  通路富集分析    │    │  PPI 网络分析   │    │  AOP 匹配分析   │        │
│  │  (按效应分组)   │    │  (网络传播)     │    │  (全路径评估)   │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│            │                       │                       │               │
│            └───────────────────────┼───────────────────────┘               │
│                                    ↓                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    Part 3: 综合毒理学评估报告                           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 核心创新点总结

| 创新点 | 技术内容 | vs DTIAM | vs tipFormer |
|--------|----------|----------|--------------|
| **创新 1：原子-残基级 BilinearAttention** | 保留 token/残基级序列表征，多头双线性注意力学习非对称交互 | DTIAM 无交互建模（特征拼接） | tipFormer 的 CrossAttention 退化为 seq_len=1 的伪序列级 |
| **创新 2：原子级可解释性** | BerMol token=原子，注意力矩阵直接映射到分子图；RDKit 官能团池化产出毒理学级别的可解释性 | DTIAM 为黑盒 AutoGluon | tipFormer 为全局向量交互，无法归因 |
| **创新 3：多源效应数据整合** | 7 个数据源（CTD + ToxCast + DTIAM + ChEMBL + IUPHAR + DGIdb + T3DB），药理学+毒理学联合训练 | DTIAM 仅 TTD 单一来源 | tipFormer 不预测效应 |
| **创新 4：跨域迁移评估** | 训练用药理学+毒理学，测试仅毒理学数据（Cold Chemical Split） | DTIAM 无跨域设计 | tipFormer 无效应任务 |
| **创新 5：多维结局关联** | 通路富集（按效应分组）+ PPI 网络 + AOP 匹配 | DTIAM 无下游分析 | tipFormer 仅通路验证 |

---

## 二、数据基础

### 2.1 Binding 任务数据

沿用已有数据集，Domain-Aware 划分策略：

| 来源 | 类型 | 数量 |
|------|------|------|
| BindingDB | 药物正例 | 13,393 |
| T3DB | 毒理正例 | 37,148 |
| ToxCast | 毒理正例 | 6,905 |
| APDB | 毒理正例 | 2,455 |
| PubMed_Filtered | 负例 | 71,652 |

**划分策略**：
- Warm Start：药物正例→全部训练集，毒理正例→70%训练/15%验证/15%测试（随机）
- Cold Chemical：药物正例→全部训练集，毒理正例→按化学品冷划分（验证/测试化学品不在训练集）

### 2.2 Effect 任务数据

#### 2.2.1 数据来源（v2 数据池）

| 来源 | 记录数 | 类型 | 特点 |
|------|--------|------|------|
| DTIAM (TTD) | 22,025 | 药物 | 极度偏向 antagonist (594:5886) |
| Toxicology (CTD + ToxCast/InVitroDB) | 17,700 | 毒理 | 核心数据 |
| ChEMBL | 14,080 | 药物 | agonist 偏多 |
| IUPHAR | 11,910 | 药物 | 较平衡 |
| DGIdb | 11,084 | 药物 | 汇聚多源标注 |
| **合计** | **76,799** | | |

#### 2.2.2 任务定义

采用与 DTIAM 相同的两个独立二分类任务：

- **Activation 任务**：给定（化学品，蛋白）对，是否存在激活效应？
  - 正样本：有 activation/agonist 标签的对
  - 负样本：随机采样的（化学品，蛋白）对
- **Inhibition 任务**：给定（化学品，蛋白）对，是否存在抑制效应？
  - 正样本：有 inhibition/antagonist 标签的对
  - 负样本：随机采样的（化学品，蛋白）对
- 正负比参考 DTIAM 的 1:10

#### 2.2.3 放弃层次化预测的原因

| 方案 | 放弃原因 |
|------|----------|
| agonist vs antagonist 直接二分类 | 75% 化学品仅有单方向效应，cold chemical 评估无意义 |
| 层次化预测（has_effect → direction） | **no_effect 负样本无法可靠构造**——无法区分"未报道交互"和"确认无效应" |
| binding-aware 负采样 | 也需要可靠的"结合但无效应"样本，同样不可得 |

两个独立二分类+随机负采样是目前最务实、数据需求最简单的方案。

#### 2.2.4 数据使用策略

```
训练集：药理学 + 毒理学数据（联合训练）
测试集：仅毒理学数据（Cold Chemical Split）
负样本：随机采样（化学品，蛋白）对
```

---

## 三、Part 1：CPI 预测模型（核心）

### 3.1 整体架构（E6 实际实现）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ToxCPI 模型架构 (E6, v4.0)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   输入                                                                      │
│   ├── 化学品: SMILES → BerMol 预提取 → (n_atoms, 768)                      │
│   └── 蛋白质: 序列 → ESM-2 预提取 → (seq_len, 1280)                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              模块 A: 投影层                                          │  │
│   │                                                                      │  │
│   │   化学品: Linear(768 → 128)        蛋白质: Linear(1280 → 128)       │  │
│   │          ↓                                   ↓                       │  │
│   │   H_atom ∈ R^(n_atoms × 128)      H_res ∈ R^(seq_len × 128)       │  │
│   │                                                                      │  │
│   │   关键特性:                                                          │  │
│   │   ├── BerMol token = 原子（Morgan radius=1 分词）                   │  │
│   │   │   → 每个 token 对应一个重原子的局部化学环境                     │  │
│   │   │   → token 索引 = 原子索引（天然一一对应）                       │  │
│   │   └── ESM-2 残基级特征                                              │  │
│   │       → 每个位置对应一个氨基酸残基                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              模块 B: 多头 BilinearAttention (4 heads)                │  │
│   │                                                                      │  │
│   │   对每个 head h:                                                     │  │
│   │   A_h[i,j] = σ(H_atom[i]ᵀ · W_h · H_res[j])                       │  │
│   │   W_h ∈ R^(d_head × d_head) 是可学习的双线性交互矩阵               │  │
│   │                                                                      │  │
│   │   向量化实现 (einsum):                                               │  │
│   │   transformed = einsum('bna, had -> bhnd', H_atom, W)               │  │
│   │   A = matmul(transformed, H_res^T)                                  │  │
│   │                                                                      │  │
│   │   与 tipFormer CrossAttention 的本质区别:                            │  │
│   │   ├── tipFormer: CLS × CLS → seq_len=1 的伪注意力                  │  │
│   │   └── ToxCPI: (n_atoms × seq_len) → 真正的原子-残基级交互          │  │
│   │                                                                      │  │
│   │   输出:                                                              │  │
│   │   ├── h_interact ∈ R^(256) — 交互特征向量                          │  │
│   │   └── A ∈ R^(n_atoms × seq_len) — 注意力矩阵（可解释性）          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              模块 C: 分类器                                          │  │
│   │                                                                      │  │
│   │   MLP: 256 → 128 → 64 → 1                                          │  │
│   │                                                                      │  │
│   │   Binding / Activation / Inhibition 任务共享编码器，各自独立分类头   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   输出                                                                      │
│   ├── P(binding) / P(activation) / P(inhibition)                           │
│   └── 注意力矩阵 A → 原子-残基交互热图 → 官能团池化可视化               │
│                                                                             │
│   模型规模: ~699K 参数                                                     │
│   训练策略: Label Smoothing + EMA + 5-Seed 集成                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块 A：预训练特征编码

#### 3.2.1 化学品编码：BerMol token 级特征

BerMol（DTIAM 提出的分子预训练模型）的分词机制是本研究可解释性的关键基础：

```python
# BerMol 分词逻辑 (bermol/utils.py)
def mol_to_sentence(mol, radius=1):
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]  # 按原子索引顺序
    atom_dic = {x: {r: None for r in range(radius+1)} for x in mol_atoms}

    for element in info:
        for atom_idx, r in info[element]:
            atom_dic[atom_idx][r] = element

    sentence = []
    for atom in atom_dic:
        if atom_dic[atom][radius]:
            sentence.append(str(atom_dic[atom][radius]))
    return sentence
```

**关键特性**：
- 每个 token 是以一个重原子为中心的 radius=1 Morgan 子结构（原子 + 一阶邻居）
- token 序列按原子索引顺序排列：**token[i] = atom[i]**
- n_tokens = n_heavy_atoms（重原子数）
- 最终输入 BERT 后：`[CLS] + atom_0 + atom_1 + ... + atom_n + [SEP]`

与 ChemBERTa 的本质区别：

| | ChemBERTa | BerMol |
|---|---|---|
| 分词方式 | BPE（按频率切分 SMILES 字符串） | Morgan radius=1（每个原子的局部化学环境） |
| token 含义 | 无化学意义的字符碎片 | 有化学意义的原子环境 |
| token 与原子对应 | 不对应 | **一一对应** |
| 可解释性 | 差 | **好** |

#### 3.2.2 蛋白质编码：ESM-2 残基级特征

预提取 ESM-2 (650M) 的残基级隐藏状态：

- 输入：氨基酸序列
- 输出：(seq_len, 1280) 的残基级特征矩阵
- 每个位置对应一个氨基酸残基，可直接用于可解释性分析

#### 3.2.3 为什么不用 ToxAlert/ToxPrint？

E1-E3 实验验证了 ToxPrint 导致严重的 shortcut learning：

| 实验 | 配置 | 门控偏好 ToxPrint | Cold AUROC |
|------|------|------------------|-----------|
| E1 | ToxPrint + ChemBERTa 门控融合 | 97.7% | 0.7435 |
| E3 | ToxPrint + BerMol 门控融合 | 98.5% | 0.7309 |
| **E4** | **纯 BerMol（去掉 ToxPrint）** | — | **0.8960** |

训练集中化学品可见时，ToxPrint 手工指纹区分力极强，模型贪婪利用这条"捷径"而完全抑制语义特征。到 Cold 场景，ToxPrint 对未见化学品泛化差，被抑制的语义特征也无法发挥作用。

**结论**：ToxPrint/ToxAlert 不进入模型，但可以在推理阶段作为 RDKit 官能团池化的知识来源，用于可解释性分析。

### 3.3 模块 B：多头 BilinearAttention（核心创新）

#### 3.3.1 为什么比 tipFormer 的 CrossAttention 更好？

tipFormer 的 CrossAttention 实际实现 (`tipFormer model.py:157-166`)：

```python
# tipFormer 的 "CrossAttention" 实际上:
comp_seq = comp_enc.unsqueeze(1)   # (B, 1, H) ← seq_len=1
prot_seq = prot_enc.unsqueeze(1)   # (B, 1, H) ← seq_len=1
comp_attn_out, _ = self.comp_attn(query=comp_seq, key=prot_seq, value=prot_seq)
# 这只是两个全局向量之间的线性变换，不是真正的残基级注意力
```

| 特性 | tipFormer CrossAttention | ToxCPI BilinearAttention |
|------|--------------------------|--------------------------|
| 化学品维度 | CLS 全局向量 (1×H) | 原子级序列 (n_atoms×H) |
| 蛋白质维度 | CLS 全局向量 (1×H) | 残基级序列 (seq_len×H) |
| 注意力矩阵 | 1×1（无意义） | n_atoms × seq_len（原子-残基级） |
| 相似度计算 | 点积 Q·K^T/√d | 双线性 H^T·W·H（可学习非对称关系） |
| 可解释性 | 无 | 注意力热图直接反映原子-残基交互强度 |

#### 3.3.2 实现细节

```python
class BilinearAttention(nn.Module):
    """多头双线性注意力，学习原子-残基非对称交互"""
    def __init__(self, atom_dim, residue_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = hidden_dim // num_heads

        # 可学习的双线性交互矩阵 (num_heads, d_head, d_head)
        self.W = nn.Parameter(torch.randn(num_heads, self.d_head, self.d_head))

        # 投影层
        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.res_proj = nn.Linear(residue_dim, hidden_dim)

    def forward(self, H_atom, H_residue, atom_mask=None, res_mask=None):
        # H_atom: (batch, n_atoms, atom_dim)
        # H_residue: (batch, seq_len, residue_dim)

        H_a = self.atom_proj(H_atom)    # (batch, n_atoms, hidden_dim)
        H_r = self.res_proj(H_residue)  # (batch, seq_len, hidden_dim)

        # 向量化多头计算
        # transformed: (batch, num_heads, n_atoms, d_head)
        transformed = torch.einsum('bna, had -> bhnd', H_a, self.W)

        # A: (batch, num_heads, n_atoms, seq_len)
        A_multi = torch.matmul(transformed, H_r.transpose(-1, -2).unsqueeze(1))

        # mask + softmax
        A = A_multi.mean(dim=1)  # 多头平均
        A_norm = F.softmax(A, dim=-1)  # 每个原子对残基的注意力分布

        # 加权聚合
        context = torch.bmm(A_norm, H_r)  # (batch, n_atoms, hidden_dim)
        h_interact = context.mean(dim=1)   # 全局池化

        return h_interact, A  # A 用于可解释性
```

### 3.4 模块 C：任务预测头

#### Binding 任务

```
共享编码器 → h_interact (256d) → MLP (256→128→64→1) → sigmoid → P(binding)
正负样本：已有 CPI 数据 vs PubMed 过滤负例
```

#### Activation 任务

```
共享编码器 → h_interact (256d) → MLP (256→128→64→1) → sigmoid → P(activation)
正样本：有 activation/agonist 标签的 (化学品, 蛋白) 对
负样本：随机采样 (化学品, 蛋白) 对，正负比 1:10
```

#### Inhibition 任务

```
共享编码器 → h_interact (256d) → MLP (256→128→64→1) → sigmoid → P(inhibition)
正样本：有 inhibition/antagonist 标签的 (化学品, 蛋白) 对
负样本：随机采样 (化学品, 蛋白) 对，正负比 1:10
```

三个任务共享化学品编码器、蛋白质编码器和 BilinearAttention 交互模块，仅分类头独立。

### 3.5 可解释性方案（核心创新之二）

#### 3.5.1 原子-残基注意力矩阵

模型推理时直接输出注意力矩阵 A ∈ R^(n_atoms × seq_len)，其中 A[i,j] 表示化学品第 i 个原子与蛋白质第 j 个残基的交互强度。

由于 BerMol 的 token=原子特性（Morgan radius=1），这个矩阵可以直接映射到：
- 化学品分子图上的具体原子
- 蛋白质序列上的具体氨基酸残基

#### 3.5.2 RDKit 官能团池化（后处理）

在推理阶段，将原子级注意力聚合到毒理学有意义的官能团级别：

```python
def functional_group_attention(A, mol, smarts_dict):
    """
    将原子级注意力聚合到官能团级别

    Args:
        A: 注意力矩阵 (n_atoms, seq_len)
        mol: RDKit Mol 对象
        smarts_dict: 官能团 SMARTS 字典
            例如 {"硝基": "[N+](=O)[O-]", "芳香胺": "[NH2]c1ccccc1", ...}

    Returns:
        group_attention: dict {官能团名: (attention_vector, atom_indices)}
    """
    group_attention = {}

    for group_name, smarts in smarts_dict.items():
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)

        for match_idx, atom_indices in enumerate(matches):
            # 直接用原子索引切片 BerMol 的注意力矩阵
            group_attn = A[list(atom_indices), :].mean(dim=0)  # (seq_len,)
            key = f"{group_name}_{match_idx}" if len(matches) > 1 else group_name
            group_attention[key] = {
                'attention': group_attn,        # 该官能团对每个残基的注意力
                'atom_indices': atom_indices,    # 涉及的原子索引
                'importance': group_attn.sum()   # 该官能团的整体重要性
            }

    return group_attention
```

**官能团知识来源**：
- ToxPrint 子结构（729 个毒理学相关 SMARTS）
- Benigni-Bossa 致癌警示结构（33 个）
- EDC 内分泌干扰物结构（50 个）
- 自定义毒理学关键基团

**可视化输出**：
- Figure A：分子图上标注原子重要性（颜色深浅）
- Figure B：蛋白质序列上标注残基重要性
- Figure C：官能团-残基交互热图（行=官能团，列=残基）

这样 ToxPrint/ToxAlert 的领域知识以"分析工具"而非"模型特征"的角色参与，避免了 shortcut learning 风险。

### 3.6 训练策略

| 组件 | 方案 |
|------|------|
| 特征提取 | 预提取 BerMol token 级 + ESM-2 残基级，冻结不微调 |
| 优化器 | AdamW (lr=1e-4, weight_decay=5e-5) |
| 正则化 | Dropout 0.3 + Label Smoothing 0.05 (Cold) / 0.02 (Warm) |
| EMA | 指数移动平均 (decay=0.999) |
| 集成 | 5-seed 概率平均（固定数据划分，不同模型初始化） |
| Batch Size | 128 |
| Early Stopping | patience=10 (Cold) / 15 (Warm) |

### 3.7 当前 Binding 任务性能

| 数据集 | 划分 | tipFormer (论文) | ToxCPI E6 集成 | 提升 |
|--------|------|----------------|----------------|------|
| tipFormer 数据 | Warm | 0.9787 | **0.9806** | +0.19% |
| tipFormer 数据 | Cold Chemical | 0.9206 | **0.9235** | +0.29% |
| 我们的数据 | Warm | 0.9663 | **0.9731** | +0.68% |
| 我们的数据 | Cold Chemical | 0.8825 | **0.9103** | +2.78% |

**四个场景全面超越 tipFormer，Cold Chemical 上领先最大。**

---

## 四、Part 2：毒理学结局关联分析

### 4.1 通路富集分析（按效应类型分组）

**创新点**：不仅做整体富集，还按激活/抑制分组分析，揭示化学品的双向调控模式。

```python
def pathway_enrichment_by_effect(predictions, threshold=0.7):
    activated_proteins = [p for p, prob in predictions['activation'] if prob > threshold]
    inhibited_proteins = [p for p, prob in predictions['inhibition'] if prob > threshold]

    results = {
        'activated': enrichr_analysis(activated_proteins, 'KEGG_2021_Human'),
        'inhibited': enrichr_analysis(inhibited_proteins, 'KEGG_2021_Human')
    }

    # 识别双向调控通路
    act_pathways = set(results['activated']['pathway'])
    inh_pathways = set(results['inhibited']['pathway'])
    results['bidirectional_pathways'] = act_pathways & inh_pathways

    return results
```

### 4.2 PPI 网络分析

（保持原方案）

### 4.3 AOP 全路径匹配与风险评估

（保持原方案）

---

## 五、训练与评估

### 5.1 数据划分策略

```
两种划分策略：

1. Warm Start（热启动）
   ├── 比例：70% / 15% / 15%
   ├── 训练/测试化学品有重叠
   └── 测试：基本学习能力

2. Cold Chemical Split（化学品冷启动）— 最重要
   ├── 测试集化学品在训练集中完全未出现
   ├── 测试：对全新环境化学品的泛化能力
   └── ES&T 审稿人最关注的场景

注意：测试集仅使用毒理学来源数据
```

### 5.2 评估指标

| 任务 | 主要指标 | 次要指标 |
|------|----------|----------|
| Binding | AUROC, AUPRC | Accuracy, F1, MCC |
| Activation | AUROC, AUPRC | Recall, Precision |
| Inhibition | AUROC, AUPRC | Recall, Precision |

### 5.3 实验设计

#### 实验 1：与现有方法对比

| 对比方法 | 来源 | 对比维度 |
|----------|------|----------|
| tipFormer | ES&T 2025 | Binding 任务（已完成，四场景全面领先） |
| DTIAM | Nat Commun 2025 | Effect 任务（Activation + Inhibition） |
| XGBoost / LightGBM | 传统 ML | 使用相同预训练特征的集成方法基线 |

#### 实验 2：消融实验

| 配置 | 说明 | 验证点 |
|------|------|--------|
| Full Model (E6) | BerMol tokens + ESM-2 residues + BilinearAttn | 基准 |
| w/o token-level | BerMol CLS + ESM-2 CLS + MLP | 验证序列级表征的价值 |
| w/o BilinearAttn | BerMol tokens + ESM-2 residues + CrossAttn | 验证双线性注意力 vs 标准注意力 |
| w/o ESM-2 residues | BerMol tokens + ESM-2 CLS | 验证残基级特征的贡献 |
| + ToxPrint fusion | 加入 ToxPrint 门控融合 | 验证 ToxPrint shortcut learning 结论 |

#### 实验 3：可解释性验证

```
├── 原子/官能团重要性验证
│   ├── 用 RDKit 提取毒理学官能团，检查注意力是否集中在已知毒性基团上
│   └── 与 ToxPrint 标注的毒性子结构对比
│
├── 残基重要性验证
│   ├── 与 PDB 共晶结构中的配体接触残基对比（4Å 内）
│   └── 与 UniProt binding site / active site 注释对比
│
└── 案例研究
    ├── BPA（双酚A）：ESR1 激活机制 — 哪些酚羟基与哪些残基交互
    ├── Chlorpyrifos：AChE 抑制机制 — 有机磷基团与活性位点丝氨酸
    ├── PFOA：PPAR 相关肝毒性 — 全氟碳链与配体结合域
    └── Atrazine：内分泌干扰机制 — 三嗪环与雄激素受体
```

---

## 六、论文写作框架

### 6.1 建议标题

**方案 1（强调可解释性）：**
```
ToxCPI: An Interpretable Deep Learning Framework for Predicting
Chemical-Protein Interaction Effects in Environmental Toxicology
```

**方案 2（强调应用）：**
```
From Molecular Interactions to Adverse Outcomes: Predicting Activation
and Inhibition Effects of Environmental Chemicals on Human Proteins
```

### 6.2 创新点表述（供审稿人）

> **Our contributions:**
>
> 1. We propose ToxCPI, a unified deep learning framework that predicts binding, activation, and inhibition effects between environmental chemicals and human proteins, extending beyond binding-only prediction (tipFormer) with application-specific toxicological evaluation.
>
> 2. We introduce **atom-residue bilinear attention** that leverages BerMol's atom-aligned tokenization (Morgan radius=1 substructures) and ESM-2 residue-level embeddings to model fine-grained molecular interactions, providing interpretable attention maps showing which chemical substructures interact with which protein regions.
>
> 3. We demonstrate a **functional group attention pooling** approach that aggregates atom-level attention weights into toxicologically meaningful groups (e.g., nitro groups, aromatic amines, EDC-related structures) using RDKit substructure matching, enabling domain-relevant mechanistic interpretation.
>
> 4. We integrate **multi-source effect data** from 7 databases (CTD, ToxCast, DTIAM, ChEMBL, IUPHAR, DGIdb, T3DB) with a cross-domain training strategy (pharmacological + toxicological joint training, toxicology-only testing), significantly expanding coverage beyond single-source approaches.
>
> 5. We develop a comprehensive **adverse outcome association module** that integrates pathway enrichment (grouped by effect type), PPI network analysis, and AOP matching to link molecular-level predictions with toxicological endpoints.

### 6.3 关键图表

| 图号 | 内容 | 要点 |
|------|------|------|
| Figure 1 | 整体框架图 | 三任务 + 原子-残基注意力 + 下游分析 |
| Figure 2 | 数据集构建与统计 | 7 个数据源整合 + 跨域训练策略 |
| Figure 3 | 模型架构对比 | ToxCPI vs tipFormer（真序列级 vs 伪序列级注意力） |
| Figure 4 | Binding 性能对比 | 四场景全面超越 tipFormer |
| Figure 5 | Effect 性能对比 | vs DTIAM，Cold Chemical 上的效应预测 |
| Figure 6 | 消融实验 | 各组件贡献验证 |
| Figure 7 | 可解释性验证 | 官能团-残基热图 + PDB 结合位点对比 |
| Figure 8 | 案例研究 | BPA/Chlorpyrifos 完整分析流程 |

---

## 七、模型迭代历程（关键技术决策记录）

| 阶段 | 实验 | 配置 | Cold AUROC | 结论 |
|------|------|------|-----------|------|
| 端到端 | v1-v4 | ToxPrint + SimpleProteinEncoder + BilinearAttn | 0.80 | 手工特征对未见化学品泛化差 |
| 融合探索 | E1-E3 | ToxPrint + 语义特征门控融合 | 0.73-0.79 | **ToxPrint shortcut learning** |
| 纯语义 | E4 | BerMol CLS + ESM-2 CLS + MLP | 0.896 | 去掉 ToxPrint 后 Cold 突破 |
| 从零训练 | E5 | BerMol CLS + SimpleProteinEncoder | 0.67 | 从零训练蛋白质编码器不可行 |
| **序列级** | **E6** | **BerMol tokens + ESM-2 residues + BilinearAttn** | **0.907** | **Cold/Warm 均超 tipFormer + 可解释性** |
| 对比学习 | v6 | E6 + CLIP 对比损失 | 0.905 | 无效，放弃 |
| **集成** | **5-seed** | **E6 × 5 概率平均** | **0.910** | **最终方案** |

---

## 八、风险评估与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|----------|
| Effect 任务深度学习不如 DTIAM 的 AutoML | 中 | 性能对比被动 | 同时实现集成方法基线；论文重心放在应用价值和可解释性 |
| 效应数据量有限（~7.7 万条） | 中 | 模型学习不充分 | 随机负采样扩充；多源数据补充 |
| BerMol token 不覆盖全部原子 | 低 | 部分原子丢失 | 验证显示绝大多数原子有 radius=1 Morgan 环境 |
| 官能团池化信息损失 | 低 | 可解释性精度下降 | 同时保留原子级和官能团级两种可视化 |
| Cold Chemical 过拟合 | 中 | 泛化性不足 | Label Smoothing + EMA + 多 seed 集成 |

---

*文档版本：v4.0*
*更新时间：2026 年 4 月 3 日*
*主要更新：效应任务改为两个独立二分类；模型架构对齐 E6 实际实现；新增原子级可解释性方案*
