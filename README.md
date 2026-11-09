# FlowHMR2VMD

将FlowHMR生成的动作数据转换为VMD格式，并将关键帧注册到当前选中模型的骨骼上。

## 依赖

- Blender 3.2 或更高版本。
- MMD Tools 插件。
- 含 numpy + scipy + torch 的 Python 3 环境（通常位于动补软件文件夹中）。

## 安装

- **脚手架开发**：在 PyCharm 中运行 `test.py`（运行配置 `test`），框架会自动打包、
  安装到 Blender 的 addons 目录并启动 Blender。
- **独立安装**：使用脚手架的 `release.py` 生成发布 zip（发布包会包含
  `common` 自动加载模块），再在 Blender → **Preferences > Add-ons >
  Install from disk** 选择该 zip。不要只压缩插件文件夹，否则缺少 `common`。
- 无论哪种方式，都需要启用 **MMD Tools**。

## 代码结构

```text
flowhmr2vmd/
├─ __init__.py           auto_load 注册入口
├─ config.py            脚手架标识
├─ operators/           operators.py：总调度
├─ panel/               panel.py：FLOWHMR2VMD_PT_retarget
├─ properties/          properties.py：FLOWHMR2VMD_Properties（Scene PointerProperty）
├─ core/                constants / mmd / smplh / geometry / solver_client / bake
├─ worker/              solve_worker.py：外部 Python 解算入口
└─ vendor/mocap/        RGB MoCap 解算模块（仅 worker 使用）
```

## 手指骨骼约束

面板中的 **展开手指骨骼约束** 会展开一个 box，里面按列显示六组基础手指的
XYZ 最小/最大角度（默认值对应左手指）：

- 親指０ / 親指１ / 親指２
- 人指１ / 人指２ / 人指３

中指、薬指、小指沿用对应人指的范围；`.R` 的 X、Y 范围交换并取反，Z 不变。
勾选最后的 **创建手指限制旋转约束** 会为 30 根手指骨骼添加
`LIMIT_ROTATION` 约束（名称 `手指旋转限制`，LOCAL 空间）；取消勾选只删除
本插件添加的这些约束，不影响其他约束。修改角度参数时，已启用的约束会即时更新。

## 输入优先级

插件会按以下顺序查找动作源：

1. **FlowHMR kit 输出**：文件夹及子文件夹中的 `final.npz`。
2. **RGB MoCap 工程缓存**：`corrected_initial_mixamo.json`（优先）或
   `initial_mixamo.json`。

工程模式使用以下文件：

- `<kind>_flow-*-rig.npy`：52 关节 4×4 矩阵。
- `<kind>_flow-*-rig-asset.npz`：`rest`、`parents`、`ends`。
- `<kind>_contacts.npz`：接触段；直接作为 `active`。
- `<kind>_correction.json`：`floor`（用于 origin）。
- SMPL-H `model.npz`（来自工程元数据）。
- `<kind>.pt`：修正后的 SMPL-H 参数；用于 betas 和 native 旋转。
  读取 `.pt` 需要外部 Python 带 torch；失败时会回退到 rig 旋转和 betas=0。

工程模式会优先复现 RGB MoCap 工程内 `corrected_initial` 的动作和接触段，
而不是另跑一遍 `final.npz`。

## 使用

1. 在 3D 视图中**选中 MMD 模型的任意对象**（网格或骨架均可）。
2. 打开 **View3D > Sidebar > FlowHMR2VMD > VMD 动作重定向**。
3. 选择包含 `final.npz` 的文件夹，或 RGB MoCap 工程的 `flowhmr` 目录；
   插件会自动向下查找。
4. 选择名称对照：
   - **RGB MoCap 原生**：严格使用 `上半身1` 和 `つま先`；模型必须两者都有。
   - **兼容写法**：用 `上半身` 代替 `上半身1`，用 `足先EX` 代替 `つま先`；适用于大多数标准 MMD 模型。
5. 点击 **执行**。

成功后会生成 Action `FlowHMR2VMD`，帧范围 1..N，帧率与源动作一致。

## 行为说明

- 脚部接触使用 `final.npz` 的 `foot_contacts > 0.7`，并按接触段把脚底支撑点钉在锚点上；接触误差通常为亚毫米级。
- 头部按面部朝向对齐，避免把 SMPL-H 头关节的静止前倾当成动作偏移（否则角色会一直低头）。
- 写入前会**临时关闭**参与映射骨骼上的 IK/付与约束，避免 MMD IK 覆盖烘焙的 FK。
- 如果 `足先EX` 位于 `足首D` 的付与链下，脚趾不会单独打键，而是跟随脚踝刚性运动；日志会给出提示。
- 动作是直接写在骨架上的关键帧，不生成 VMD；需要 VMD 可再用 MMD Tools 导出。

## 许可

MIT License。部分算法与常量改编自
[RGB MoCap](https://github.com/toyxyz/toyxyz_RGBMocap)（MIT）。
MMD Tools 为 GPL-3.0-or-later，未随本插件分发。
详见 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。
