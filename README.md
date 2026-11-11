# FlowHMR2VMD

> **本项目已停止开发。** 不再新增功能，也不再修复问题。

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

## 许可

MIT License。部分算法与常量改编自
[RGB MoCap](https://github.com/toyxyz/toyxyz_RGBMocap)（MIT）。
MMD Tools 为 GPL-3.0-or-later，未随本插件分发。
详见 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。
