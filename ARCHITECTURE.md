# CuspNet 项目架构规格说明书

> 本文档记录项目当前的实际架构与实现状态，与代码保持严格对齐。
> 理论依据详见 [README.md](README.md)，本文只关注"怎么建"和"建到哪了"。

---

## 1. 文件架构树

```
mental-health-system/
├── README.md                            # 项目说明
├── ARCHITECTURE.md                      # 本文件
│
└── backend/
    ├── .env                             # 环境变量
    ├── requirements.txt                 # 依赖清单
    ├── run_exp1.py                      # 实验1运行入口
    ├── run_exp2.py                      # 实验2运行入口
    ├── run_exp4.py                      # 实验4运行入口
    ├── run_exp5.py                      # 实验5运行入口
    │
    └── app/                             # 所有 Python 代码
        ├── __init__.py                  # 顶层导出（全模块）
        ├── main.py                      # FastAPI 入口 + API 路由
        │
        ├── core/                        # 基础设施层
        │   ├── __init__.py
        │   ├── config.py                # 全局配置常量 (Settings)
        │   ├── database.py              # SQLAlchemy 引擎 + 会话
        │   ├── models.py                # ORM 模型 (User, Student, CuspNetRecord)
        │   └── schemas.py               # Pydantic 请求/响应模型
        │
        ├── cuspnet/                     # ★ 核心三层架构
        │   ├── __init__.py              # 导出4个核心类 + 2个工具函数
        │   ├── utils.py                 # 纯数学工具函数
        │   ├── layer1_causal.py         # Layer 1: 因果网络发现
        │   ├── layer2_dynamics.py       # Layer 2: Cusp 分岔动力学
        │   ├── layer3_llm.py            # Layer 3: 多步反思 LLM
        │   └── cuspnet_engine.py        # 三层融合引擎
        │
        ├── visualization/               # 可视化
        │   ├── __init__.py
        │   ├── causal_graph_vis.py      # 因果 DAG 图（返回前端可渲染JSON）
        │   ├── potential_vis.py         # 势函数曲面
        │   └── network_vis.py           # 中心性热力图
        │
        ├── data/                        # 数据管道
        │   ├── __init__.py
        │   ├── loaders.py               # 5个数据集加载器
        │   ├── preprocess.py            # 标准化 + 缺失值
        │   ├── synthetic.py             # Cusp ODE 合成数据生成
        │   └── raw/                     # 原始数据集
        │       ├── sachs_data.csv       # Sachs蛋白质网络 (公开)
        │       ├── DPQ_J.XPT            # NHANES DPQ原始 (公开)
        │       ├── nhanes_dpq.csv       # NHANES DPQ派生 (公开)
        │       ├── kossakowski_esm.csv  # ESM追踪数据 (合成)
        │       └── daic_woz.csv         # DAIC-WOZ访谈 (合成)
        │
        └── experiments/                 # 实验脚本
            ├── __init__.py              # 导出全部实验函数
            ├── exp1_causal_discovery.py # 实验1: 因果发现
            ├── exp2_cusp_fitting.py     # 实验2: Cusp拟合
            ├── exp3_resilience_prediction.py # 实验3: 韧性预测（已实现，未独立运行）
            ├── exp4_llm_appraisal.py    # 实验4: LLM评估链
            ├── exp5_end_to_end.py       # 实验5: 端到端预测
            ├── ablation_study.py        # 消融实验（已实现，未独立运行）
            └── baselines/
                ├── __init__.py
                ├── pc_algorithm.py
                ├── ges_algorithm.py
                ├── notears_baseline.py
                └── ml_baselines.py
```

**启动方式**：`uvicorn app.main:app --reload`

---

## 2. 后端基础设施

### 2.1 `app/core/config.py`

**职责**：集中管理所有可配置参数，通过 `.env` 覆盖

**类**：`Settings(BaseSettings)`

| 字段 | 类型 | 实际默认值 | 用途 |
|------|------|-----------|------|
| `DATABASE_URL` | str | `sqlite:///./mental_health.db` | 数据库连接 |
| `LLM_MODEL_NAME` | str | `D:\Models\huggingface\Qwen3.5-2B` | LLM 模型（本地路径） |
| `LLM_DEVICE` | str | `cuda` | 推理设备 |
| `LLM_LOAD_IN_4BIT` | bool | **False** | 4-bit 量化（2B模型无需量化） |
| `LLM_MAX_NEW_TOKENS` | int | 2048 | 最大生成长度（config层） |
| `LLM_TEMPERATURE` | float | 0.3 | 采样温度 |
| `EBIC_GAMMA` | float | 0.5 | EBIC 扩展参数 γ |
| `SCORE_THRESHOLD` | float | 0.01 | 偏相关阈值 |
| `CUSP_THETA_BIFURCATION` | float | 0.5 | 分岔阈值 θ |
| `ALLOCATION_LAMBDA_A` | float | 0.3 | 参数分配系数 λ₁ |
| `ALLOCATION_LAMBDA_B` | float | 0.3 | 参数分配系数 λ₂ |
| `ALLOCATION_LAMBDA_C` | float | 0.2 | 参数分配系数 λ₃ |
| `STATE_DRIFT_ETA` | float | 0.1 | 状态漂移步长 η |

> **注意**：`layer3_llm.py` 构造函数中 `max_new_tokens` 默认值为 512，与 config 层的 2048 不同。实际运行时由 config 传入 2048。

### 2.2 `app/core/database.py`

**职责**：SQLAlchemy 引擎 + 会话工厂

**导出**：
- `engine` = `create_engine(DATABASE_URL, connect_args={"check_same_thread": False})`
- `SessionLocal` = `sessionmaker(engine)`
- `get_db()` → FastAPI 依赖注入的生成器

### 2.3 `app/core/models.py`

**职责**：3 个 ORM 模型

| 模型 | 表名 | 关键字段 |
|------|------|----------|
| `User` | users | id, username, hashed_password, role(counselor/admin), is_active, created_at |
| `Student` | students | id, student_id, name, gender, age, grade, major, college, gpa, attendance_rate, **phq9_scores(JSON)**, **gad7_scores(JSON)**, **pss10_scores(JSON)**, **cdrisc_scores(JSON)**, **mssp_scores(JSON)**, created_at, updated_at |
| `CuspNetRecord` | cuspnet_records | id, student_id(FK), risk_score, risk_level, **global_a/b/c**, resilience_reserve, critical_distance, tipping_point_warning, causal_adjacency(JSON), centrality_ranking(JSON), positive_feedback_loops(JSON), attractor_states(JSON), llm_primary/secondary/distortions/intervention/explanation(JSON), model_version, assessed_at |

**关系**：Student 1→N CuspNetRecord（cascade delete）

### 2.4 `app/core/schemas.py`

**职责**：Pydantic 请求/响应模型

| 模型 | 用途 | 关键字段 |
|------|------|----------|
| `RiskLevel` | 枚举 | low / medium / high / critical |
| `CuspNetAssessmentRequest` | 请求 | student_id, text_input?, behavior_data? |
| `CausalNetworkResult` | Layer1输出 | precision_matrix, partial_correlation, causal_adjacency, topological_order, centrality_ranking, bridge_centrality, bridge_symptoms, positive_feedback_loops, variable_names |
| `DynamicsResult` | Layer2输出 | global_a/b/c, local_params[], attractor_states, resilience_reserve, critical_distance, tipping_point_warning, potential_function, drift_prediction?, simulation? |
| `LLMAppraisalResult` | Layer3输出 | primary_appraisal, secondary_appraisal, reappraisal, cognitive_distortions[], cusp_proxies{a/b/c_proxy}, explanation, intervention |
| `CuspNetAssessmentResponse` | 总响应 | risk_score, risk_level, causal_network, dynamics, llm_appraisal?, model_version, assessed_at |
| `StudentBrief` | 学生摘要 | id, student_id, name, risk_level? |

### 2.5 `app/main.py`

**职责**：FastAPI 应用 + API 路由

**关键端点**：

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/assess` | 接收 CuspNetAssessmentRequest → 调用 CuspNetEngine.assess() → 返回 CuspNetAssessmentResponse |
| GET | `/api/students/{id}/history` | 获取学生历史评估记录 |
| GET | `/api/visualization/potential` | 返回势函数数据 (x_range + V)，支持 a/b/c 查询参数 |
| GET | `/api/visualization/causal-graph` | 返回因果 DAG 数据 + 中心性热力图 |
| GET | `/api/visualization/simulation` | 返回 ODE 仿真轨迹数据 |
| GET | `/health` | 健康检查 |

**核心逻辑**：
- `/api/assess` 中实例化全局 `CuspNetEngine`，将请求中的问卷数据组装为 numpy 数组后调用 `engine.assess()`
- 单样本时（N=1）走快速路径：直接计算 Cusp 参数，跳过 Layer1 的 EBICglasso（需要 N≥2）
- 多样本时走完整三层流程

**⚠ FastAPI 异步 + 本地 LLM 线程阻塞约束**：
- FastAPI 是异步并发的，而本地 LLM 的 `model.generate()` 是同步阻塞调用
- **已使用 `run_in_threadpool`** 将 LLM 推理放入后台线程池
- Layer1 和 Layer2 的计算量较小（EBICglasso + 解析求解），可直接同步调用
- 仅 Layer3 的 LLM 推理需要异步化

---

## 3. CuspNet 核心模块

### 3.1 `app/cuspnet/utils.py` — 纯数学工具

**职责**：无状态纯函数，供 Layer1/2/Engine 调用

| 函数 | 公式 | 输入 | 输出 |
|------|------|------|------|
| `sigmoid(x, τ, k)` | `1/(1+exp(-k(x-τ)))` | x:ndarray, τ:0.5, k:10.0 | ndarray |
| `solve_cubic(a, b, c)` | `cx³-bx-a=0` 的实根（numpy.roots） | a,b,c:float | List[float] |
| `compute_potential(x, a, b, c)` | `V=-ax-(b/2)x²+(c/4)x⁴` | x:ndarray, a,b,c:float | ndarray |
| `find_fixed_points(a, b, c)` | 调用 solve_cubic | a,b,c:float | List[float] |
| `classify_fixed_points(roots, b, c)` | `V''=-b+3cx²`, V''>0→stable | roots:List, b,c:float | List[Dict{value, stability, second_deriv}] |
| `compute_resilience_reserve(a, b, c)` | `ΔV=V(saddle)-V(healthy)` | a,b,c:float | float (inf if no bistability) |
| `compute_critical_distance(b, b_crit=0)` | `|b|` (b_crit=0时) | b:float | float |
| `detect_positive_feedback_loops(A, max_depth=5)` | DFS 找有向环 | A:ndarray(p×p) | List[List[int]] |
| `compute_loop_strength(A, loop)` | `∏|Aᵢⱼ|^(1/L)` 几何平均 | A:ndarray, loop:List[int] | float |

### 3.2 `app/cuspnet/layer1_causal.py` — Layer 1: 因果网络发现

**类**：`CausalDiscoveryLayer`

**构造参数**：`ebic_gamma=0.5`, `score_threshold=0.1`

**主方法**：`fit(X: ndarray(N×p), variable_names: List[str]) → Dict`

**内部流程**：

| 步骤 | 方法 | 实际算法 | 输出 |
|------|------|---------|------|
| 1.1 | `_ebic_glasso(X)` | **rpy2 → R qgraph::EBICglasso**（备选：Python EBIC搜索） | precision_matrix, partial_corr |
| 1.2 | `_score_algorithm(X)` | **GES（主） + BIC exact search（备） + PC+GES（兜底）** | topological_order, score_adj, score_skeleton |
| 1.3 | `_theory_constrained_orientation(partial_corr, topo_order, names, score_adj, score_skeleton)` | GES骨架 + EBICglasso权重 + Borsboom约束定向 | causal_adjacency: ndarray(p×p) |
| 1.4 | `_compute_centrality(A, names)` | 预期影响(EI) + 桥接中心性(跨社区) | centrality_ranking, bridge_centrality |

**Step 1.1 EBICglasso 实现细节**：
- 优先使用 rpy2 调用 R 的 `qgraph::EBICglasso()`
- R 环境自动检测：通过 Windows 注册表 `SOFTWARE\R-core\R64` 或候选路径查找 R_HOME
- rpy2 调用方式：`qgraph.EBICglasso(R, n, gamma=self.ebic_gamma, nlambda=100, lambda_min_ratio=0.01)`
- R 不可用时自动降级为 Python 实现：手动搜索 100 个 lambda 值，对每个调用 `sklearn.covariance.GraphicalLasso` 并计算 EBIC，取最小 EBIC 对应的精度矩阵

**Step 1.2 因果拓扑排序实现细节**：
- **当前实际使用 GES（Greedy Equivalence Search）而非 SCORE**
- 原因：当前安装的 `causal-learn` 版本不含独立的 SCORE API
- 实现策略（三级降级）：
  1. `_score_via_causal_learn()`：尝试 GES → 提取 CPDAG → 拓扑排序；若 GES 失败则尝试 BIC exact search
  2. `_score_fallback_pc_ges()`：GES → PC 的降级组合
  3. 兜底返回 `list(range(p))`（无序）
- GES 返回的 CPDAG 中：`G.graph[j,i]==1 && G.graph[i,j]==-1` 表示 i→j 有向边；`G.graph[i,j]==-1 && G.graph[j,i]==-1` 表示 i-j 无向边

**Step 1.3 理论约束定向实现细节**：
- 合并 GES 骨架和 EBICglasso 偏相关矩阵构建联合骨架
- 80th percentile 阈值过滤弱边
- GES 有向边优先保留（如果偏相关 > score_threshold）
- 未定向边按拓扑序方向定向（避免环）
- Borsboom 约束字典强制覆盖已知因果方向

**Borsboom 理论约束字典**（硬编码的已知因果方向）：
- insomnia → fatigue, fatigue → concentration, anhedonia → depressed_mood
- worry → restlessness, stress → anxiety, anxiety → insomnia, social_isolation → depressed_mood

**返回 Dict 字段**：precision_matrix, partial_correlation, causal_adjacency, topological_order, centrality_ranking, bridge_centrality, bridge_symptoms, positive_feedback_loops (含loop+strength), variable_names

### 3.3 `app/cuspnet/layer2_dynamics.py` — Layer 2: Cusp 分岔动力学

**类**：`CuspDynamicsLayer`

**构造参数**：`theta_bifurcation=0.5`, `lambda_a=0.3`, `lambda_b=0.3`, `lambda_c=0.2`, `drift_eta=0.1`, `tau_threshold=0.5`, `coupling_k=10.0`

**主方法**：`full_analysis(pss10, cdrisc, mspss, cognitive_reappraisal, centrality, bridge, causal_adjacency, x0?, t_span?) → Dict`

**内部步骤**：

| 步骤 | 方法 | 公式 | 说明 |
|------|------|------|------|
| 2.1a | `estimate_global_params(pss10, cdrisc, mspss, cognitive_reappraisal)` | `a=norm(PSS)-norm(CD-RISC)`, `b=norm(CD-RISC)×norm(MSPSS)-θ`, `c=norm(MSPSS)×reappraisal` | 量表原始分除以满分归一化：PSS/50, CD-RISC/40, MSPSS/84 |
| 2.1b | `allocate_params(a,b,c, centrality, bridge)` | `aᵢ=a·(1+λ₁·centᵢ_norm)`, `bᵢ=b·(1-λ₂·centᵢ_norm)`, `cᵢ=c·(1+λ₃·bridgeᵢ_norm)` | centrality/bridge 先归一化到[0,1]（除以最大值） |
| 2.2 | `analyze_attractors(a,b,c)` | 三次方程求根 → classify | 返回 fixed_points[], is_bistable, num_stable/unstable |
| 2.3 | `compute_resilience(a,b,c)` | `ΔV=V(saddle)-V(healthy)` | 调用 utils，无双稳态时返回 inf |
| 2.4 | `simulate_network_ode(x0, a_local, b_local, c_local, A, t_span)` | `dxᵢ/dt=aᵢ+bᵢxᵢ-cᵢxᵢ³+ΣⱼAᵢⱼ·σ(xⱼ-τⱼ)` | `solve_ivp(method='BDF')`，失败时降级为解耦 |
| 2.5 | `compute_tipping_warning(b)` | `critical_distance=|b|` (b_crit=0), warning if <0.15 | 临界转变预警 |
| 闭环 | `state_dependent_drift(a_current, delta_v, eta)` | `a(t+1)=a(t)+η/|ΔV|·sign(ΔV→0)` | 状态依赖参数漂移 |

**返回 Dict 字段**：global_a/b/c, local_params[], attractor_states, resilience_reserve, critical_distance, tipping_point_warning, potential_function{x_range, V, stable/unstable_fixed_points?}, drift_prediction?, simulation?

**ODE 求解器**：
- 使用 `scipy.integrate.solve_ivp(method='BDF')`，rtol=1e-6, atol=1e-8
- 求解失败时降级为解耦模式（忽略耦合项）

### 3.4 `app/cuspnet/layer3_llm.py` — Layer 3: 多步反思 LLM

**类**：`LazarusAppraisalChain`

**构造参数**：`model_name="D:\Models\huggingface\Qwen3.5-2B"`, `device="cuda"`, `load_in_4bit=True`, `max_new_tokens=512`, `temperature=0.3`

> **注意**：config.py 中 `LLM_LOAD_IN_4BIT` 默认为 False，实际由 config 传入。2B 模型在 GPU 显存足够时无需量化。

**模型加载**：transformers AutoTokenizer + AutoModelForCausalLM
- 4-bit 量化：BitsAndBytesConfig(nf4, double_quant, bfloat16 compute dtype)
- 非量化：torch_dtype=bfloat16, device_map="auto"
- 懒加载：首次调用 `_ensure_model()` 时才加载模型
- Warmup：首次加载后执行一次单 token 生成预热

**主方法**：`full_chain(text, causal_info?, dynamics_info?) → Dict`

**5步反思流**：

| 步骤 | 方法 | 输入 | 输出 | 实现方式 |
|------|------|------|------|---------|
| 3.1 | `primary_appraisal(text)` | 文本 | {threat_identified, threat_type, threat_intensity(1-10), threat_narrative, primary_appraisal_score(1-10)} | LLM生成 + JSON解析 |
| 3.2 | `secondary_appraisal(text, primary)` | 文本+3.1输出 | {coping_resources[], coping_efficacy(1-10), social_support_perceived(1-10), resource_narrative, secondary_appraisal_score(1-10)} | LLM生成 + JSON解析 |
| 3.3 | `reappraisal(primary, secondary, text)` | 3.1+3.2 | {rollback_needed, rollback_reason, corrected_primary, corrected_secondary, lazarus_consistency, expected_stress} | **LLM生成**（非确定性逻辑） |
| 3.4 | `detect_distortions(text, corrected_primary, corrected_secondary)` | 文本+修正后评价 | [{type, severity(1-5), evidence}] | LLM生成 + JSON解析 |
| 3.5 | `integrate(primary, secondary, reappraisal, distortions, causal_info?, dynamics_info?)` | 全部 | {cusp_proxies{a/b/c_proxy}, explanation, intervention} | LLM生成 + JSON解析 |

> **与原始设计的差异**：Step 3.3 Reappraisal 原设计为"确定性逻辑（不调用LLM）"，实际实现为 LLM 生成。LLM 被要求审视初/次级评估是否存在认知偏差并给出校正分数。这提供了更灵活的反思能力，但引入了 LLM 输出的不确定性。

**代理变量公式**（在 integrate 步骤中由 LLM 输出）：
- `a_proxy = (P - S) / 10`
- `b_proxy = (S × Social) / 100 - 0.5`
- `c_proxy = Social × (11 - |distortions|) / 100`

**LLM Prompt 设计**：
- 每个 Agent 的 prompt 要求 JSON 格式输出
- 支持中英文自动检测（`_detect_language`），根据语言选择对应 prompt
- JSON 解析失败时返回默认值（score=5）
- 使用 `apply_chat_template` 构建 Qwen 格式的对话输入
- `enable_thinking=False` 关闭 Qwen3.5 的思维链模式

**返回 Dict 字段**：primary_appraisal, secondary_appraisal, reappraisal, cognitive_distortions, cusp_proxies, explanation, intervention

### 3.5 `app/cuspnet/cuspnet_engine.py` — 三层融合引擎

**类**：`CuspNetEngine`

**构造参数**：`config: Dict`（透传给三层）

**主方法**：`assess(questionnaire_data, variable_names, pss10, cdrisc, mspss, cognitive_reappraisal=0.5, text_input=None) → Dict`

**流程**：

```
1. Layer1.fit(questionnaire_data, variable_names) → l1_result
   - 提取 centrality_ranking, bridge_centrality

2. Layer2.full_analysis(pss10, cdrisc, mspss, ..., centrality, bridge, causal_adjacency) → l2_result

3. if text_input:
   Layer3.full_chain(text, causal_info, dynamics_info) → l3_result
   - 代理变量融合：
     - 有历史记录(≥3次)：Z-score 标准化后 0.5/0.5 加权融合，再反标准化
     - 无历史记录：Min-Max 归一化到 [0,1] 后 0.5/0.5 加权融合
     - 理论范围：a∈[-1,1], b∈[-0.5,1.5], c∈[0,2]；代理范围：a∈[-1,1], b∈[-1,0.5], c∈[0,1.1]
   - 融合后重新计算吸引子、韧性储备、临界距离
   - 状态依赖漂移预测（ΔV < 1.0 时触发）
   - 重新分配局部参数和ODE仿真

4. 风险评分 = _compute_risk_score(l2_result)
   - 双稳态 → +0.3
   - ΔV < 0.1 → +0.4; ΔV < 0.5 → +0.2
   - 临界距离 < 0.15 → +0.3; < 0.3 → +0.1
   - 总分 min(1.0, score)

5. 风险等级 = _classify_risk(score, tipping_warning)
   - critical (tipping warning) / high (≥0.7) / medium (≥0.4) / low
```

**返回 Dict 字段**：risk_score, risk_level, causal_network, dynamics, llm_appraisal, model_version

### 3.6 `app/cuspnet/__init__.py`

导出：`CausalDiscoveryLayer`, `CuspDynamicsLayer`, `LazarusAppraisalChain`, `CuspNetEngine`, `compute_resilience_reserve`, `compute_potential`

---

## 4. 数据管道

### 4.1 `app/data/loaders.py`

**5个加载器类**：

| 类 | 数据集 | load() 返回 | 关键字段 |
|------|------|------|------|
| `SachsLoader` | Sachs蛋白质网络 | (X:ndarray(853×11), adj_true:ndarray(11×11), var_names:list) | 11个蛋白变量，内置29条已知因果边 |
| `NHANESLoader` | NHANES心理健康 | (X:ndarray(N×10), var_names:list) | 10个DPQ列（含functional_impairment），PHQ-9+功能损害 |
| `KossakowskiLoader` | ESM高频追踪 | DataFrame | stress, resilience, social_support, mood 列 |
| `DAICWOZLoader` | DAIC-WOZ访谈 | Dict{transcripts:Dict[id→str], labels:Dict[id→int]} | 转录稿+评分 |
| `StudentLifeLoader` | StudentLife | Dict{phq9_series, stress_series, sensor_data} | 支持目录和CSV两种格式 |

**NHANESLoader 特殊处理**：
- 10个DPQ列对应10个标签（含 functional_impairment）
- 值 7/9 视为缺失值（替换为 NaN）
- 极小值（<1e-10）替换为 0
- 至少保留一半非缺失列的行
- 缺失值用列均值填充

**数据目录**：`app/data/raw/`

### 4.2 `app/data/preprocess.py`

| 函数 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `preprocess_questionnaire(X, handle_missing="mean")` | 均值填充NaN + StandardScaler | ndarray | (X_scaled, meta_dict{mean, std}) |
| `extract_cusp_params_from_student(student_data:dict)` | 从原始问卷分提取归一化参数 | dict{pss10:[], cdrisc:[], mspss:[]} | dict{pss10_norm, cdrisc_norm, mspss_norm} |

### 4.3 `app/data/synthetic.py`

| 函数 | 功能 | 关键参数 |
|------|------|----------|
| `generate_cusp_synthetic(n_samples=500, n_variables=9, ...)` | 生成Cusp ODE合成数据 | 返回 (X:ndarray, A:ndarray真实DAG, var_names:list) |

**生成逻辑**：
1. 随机生成下三角DAG邻接矩阵A（边概率0.3，权重0.1-0.5）
2. 为每个变量随机采样 a,b,c 参数
3. 对每个变量迭代100步 `x += 0.01·(a+bx-cx³+coupling)` 达到稳态
4. 加高斯噪声

---

## 5. 可视化模块

### 5.1 `app/visualization/causal_graph_vis.py`

**函数**：`plot_causal_dag(adjacency, variable_names, centrality_ranking, bridge_symptoms?) → Dict`

**输出**：前端可渲染的图数据 `{nodes: [{id, label, centrality, centrality_normalized, size, is_bridge, color}], edges: [{source, target, weight, weight_normalized, sign}]}`

**规则**：节点大小 ∝ expected_influence（归一化后映射到10-40），桥接症状红色(#e74c3c)，普通节点蓝色(#3498db)

### 5.2 `app/visualization/potential_vis.py`

**函数**：`plot_potential_surface(a, b, c, x_range?, current_state?) → Dict`

**输出**：`{x: list, V: list, fixed_points: [{x, V, stability}], current_state?, current_state_potential?}`

**逻辑**：调用 utils.compute_potential + find_fixed_points + classify_fixed_points；如果传入 current_state，额外计算当前状态的势函数

### 5.3 `app/visualization/network_vis.py`

**函数**：`plot_centrality_heatmap(adjacency, centrality_ranking, variable_names?) → Dict`

**输出**：`{matrix, heatmap_normalized, labels, centrality_values, centrality_normalized}`

**逻辑**：对角线为中心性值，非对角线为|Aᵢⱼ|，归一化到[0,1]

---

## 6. 实验模块

### 6.1 实验1：因果发现（exp1_causal_discovery）

**数据集**：Sachs（有ground truth DAG）

**对比方法**：
- CuspNet（EBICglasso + GES + Borsboom约束）
- PC算法
- GES算法
- NOTEARS算法
- EBICGLASSO_ONLY（仅偏相关网络）
- SCORE_ONLY（仅GES骨架）

**评估指标**：Precision, Recall, F1, SHD, SID

**运行入口**：`python run_exp1.py`

### 6.2 实验2：Cusp模型拟合（exp2_cusp_fitting）

**数据集**：Kossakowski ESM（合成）

**对比模型**：Cusp vs 线性 vs 逻辑斯蒂

**评估指标**：AIC, BIC, R², 预测准确率

**运行入口**：`python run_exp2.py`

### 6.3 实验3：韧性预测（exp3_resilience_prediction）

**数据集**：Kossakowski ESM / StudentLife

**方法**：滑动窗口计算 ΔV，预测临界转变

**评估指标**：ΔV 与状态变化的关联、临界预警准确率

### 6.4 实验4：LLM评估链（exp4_llm_appraisal）

**数据集**：DAIC-WOZ（合成）

**对比**：完整Lazarus链 vs 仅初级评估

**评估指标**：威胁识别准确率、抑郁水平识别准确率

**运行入口**：`python run_exp4.py`

### 6.5 实验5：端到端抑郁预测（exp5_end_to_end）

**数据集**：NHANES

**对比方法**：
- CuspNet独立预测（Cusp动力学 + logistic风险转换）
- Random Forest
- XGBoost
- CuspNet + RF（因果特征增强）
- CuspNet + XGBoost（因果特征增强）

**评估指标**：AUC-ROC, Accuracy, F1, Precision, Recall

**运行入口**：`python run_exp5.py`

### 6.6 消融实验（ablation_study）

**配置**：full / no_score / no_theory / no_cusp / no_a_embedding / no_llm / no_lazarus_reflection

---

## 7. 当前实验进度与结果

> 以下为已完成的实验结果，数据真实，未经调整。

### 实验1：因果发现（Sachs数据集）

| 方法 | Precision | Recall | F1 | SHD |
|------|-----------|--------|-----|-----|
| CuspNet | 0.25 | 0.5 | **0.333** | - |
| PC | - | - | 较低 | - |
| GES | - | - | 较低 | - |
| NOTEARS | - | - | 0.222 | - |

**结论**：CuspNet（EBICglasso + GES + Borsboom约束）在 F1 上优于各基线，理论约束定向有效提升了因果发现精度。

### 实验2：Cusp模型拟合（合成ESM数据）

| 模型 | AIC | BIC | R² | 预测准确率 |
|------|-----|-----|-----|-----------|
| Cusp | 最优 | 最优 | 0.258 | 94.87% |
| 线性 | 较差 | 较差 | - | - |
| 逻辑斯蒂 | 较差 | 较差 | - | - |

**结论**：Cusp模型在AIC/BIC上显著优于线性和逻辑斯蒂模型，验证了尖点动力学对心理健康数据的拟合优势。

### 实验4：LLM评估链（合成DAIC-WOZ数据）

| 方法 | 整体准确率 | 抑郁水平识别准确率 |
|------|-----------|-------------------|
| 完整Lazarus链 | 提升40% | 40% |
| 仅初级评估 | 基线 | 0% |

**结论**：Lazarus完整链（初级+次级+重评+认知扭曲）相比仅用初级评估，整体准确率提升40%，抑郁水平识别从0提升到40%。

### 实验5：端到端抑郁预测（NHANES数据集）

| 方法 | AUC-ROC | Accuracy |
|------|---------|----------|
| XGBoost | 0.8856 | - |
| CuspNet独立 | 0.8805 | - |
| Random Forest | 较低 | - |
| CuspNet + XGBoost | 最优 | - |

**结论**：CuspNet独立预测AUC达到0.8805，接近XGBoost的0.8856。因果特征增强后CuspNet+XGBoost达到最优。

### 实验3：韧性预测

尚未独立运行，代码已实现。

### 消融实验

尚未独立运行，代码已实现。

---

## 8. 已知差异与待改进项

| 项目 | 原始设计 | 当前实现 | 状态 |
|------|---------|---------|------|
| 因果拓扑排序 | SCORE算法 | GES + BIC exact search | ✅ 已实现，因causal-learn版本不含SCORE API |
| Reappraisal | 确定性逻辑（不调用LLM） | LLM生成 | ✅ 已实现，提供更灵活的反思 |
| LLM模型 | Qwen2.5-7B-Instruct | Qwen3.5-2B（本地） | ✅ 已适配，受显存限制 |
| 4-bit量化 | 默认开启 | 默认关闭 | ✅ 2B模型无需量化 |
| score_threshold默认值 | config.py=0.01, layer1构造函数=0.1 | Engine传入config值0.01 | ✅ 以config为准，layer1构造函数默认值仅用于独立测试 |
| NHANES PHQ标签 | 9个 | 10个（含functional_impairment） | ✅ 已修正 |
| 实验3运行 | - | 未独立运行 | ⏳ 代码已实现 |
| 消融实验运行 | - | 未独立运行 | ⏳ 代码已实现 |
| 前端可视化 | 旧版React | 已删除，待重建 | 📋 计划前后端分离重建 |
