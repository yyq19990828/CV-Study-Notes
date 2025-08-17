# 🎯 计算机视觉学习笔记

[![GitHub stars](https://img.shields.io/github/stars/username/CV-Study-Notes?style=social)](https://github.com/username/CV-Study-Notes)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Generated%20with-Claude%20Code-blue)](https://claude.ai/code)

> 专门研究目标检测算法的学习仓库，包含各种SOTA算法的官方代码实现和详细的中文技术分析文档。

## 📚 项目概述

本仓库致力于深入理解和分析计算机视觉领域的目标检测算法，特别是近年来的突破性工作。每个算法文件夹都包含：

- 🔬 **官方代码实现** - 作为Git子模块引入
- 📖 **详细技术分析** - 中文技术报告，包含核心原理、代码解析和实验结果
- 📊 **流程图可视化** - Mermaid格式的架构图和流程图

## 🚀 快速开始

### 1. 克隆仓库

```bash
# 克隆主仓库
git clone https://github.com/username/CV-Study-Notes.git
cd CV-Study-Notes
```

### 2. 初始化所有子模块

```bash
# 初始化并更新所有子模块
git submodule update --init --recursive

# 或者在克隆时直接拉取子模块
git clone --recurse-submodules https://github.com/username/CV-Study-Notes.git
```

### 3. 更新子模块到最新版本

```bash
# 更新所有子模块到最新提交
git submodule update --remote --recursive

# 或者逐个更新子模块
git submodule foreach git pull origin main
```

## 📖 技术文档归档

### 目标检测
- **[[2019]FCOS](FCOS/[2019]FCOS技术报告.md)** - 全卷积单阶段目标检测器
- **[[2021]YOLOX](YOLOX/[2021]YOLOX技术报告.md)** - 超越YOLO系列的无锚框检测器
- **[[2021]TOOD](TOOD/[2021]TOOD技术报告.md)** - 任务对齐单阶段目标检测器
- **[[2022]DINO](DINO[IDEA-端到端检测]/[2022]DINO.md)** - 端到端目标检测算法
- **[[2023]YOLOv6](YOLOv6/[2023]YOLOv6技术报告.md)** - 工业级实时目标检测框架

### 自监督学习
- **[[2021]DINO自监督学习](DINOv1/[2021]DINO_自监督学习.md)** - 自监督视觉Transformer
- **[[2024]DINOv2](DINOv2/[2024]DINOv2.md)** - DINO自监督学习改进版


## 📊 性能对比

| 算法 | 年份 | 主要创新 | COCO mAP | 特点 |
|------|------|----------|----------|------|
| **FCOS** | 2019 | 无锚框+中心度 | 44.7% | 首个实用无锚框检测器 |
| **TOOD** | 2021 | 任务对齐+T-Head | 51.1% | 解决分类定位错位问题 |
| **YOLOX** | 2021 | 解耦头+SimOTA | 51.5% | 超越YOLO系列性能 |
| **YOLOv6** | 2022-2023 | EfficientRep+BiC+AAT | 57.2% | 工业级实时检测框架 |
| **DINO** | 2022 | 对比去噪+混合查询 | 63.3% | 端到端检测新SOTA |
| **DINOv1** | 2021 | 自监督学习 | - | 视觉Transformer预训练 |
| **DINOv2** | 2024 | 改进自监督 | - | 更强的视觉特征表示 |

## ❓ 常见问题

### Q: 子模块下载失败怎么办？

A: 检查网络连接，可以尝试使用代理或镜像源：

```bash
# 使用GitHub镜像（国内用户）
git config --global url."https://github.com.cnpmjs.org/".insteadOf "https://github.com/"

# 或使用SSH方式
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

### Q: 如何只下载特定算法的代码？

A: 可以选择性初始化子模块：

```bash
# 只初始化YOLOX子模块
git submodule update --init YOLOX/YOLOX

# 只初始化FCOS子模块  
git submodule update --init FCOS/FCOS

# 只初始化YOLOv6子模块
git submodule update --init YOLOv6/YOLOv6
```

### Q: 技术文档中的代码引用如何验证？

A: 所有代码引用都包含具体的文件路径和行号，可以直接定位到源码：

```bash
# 例如查看 yolox/models/yolo_head.py:123
cd YOLOX/YOLOX
vim yolox/models/yolo_head.py +123
```

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 🙏 致谢

- 感谢各算法原作者提供的开源实现
- 感谢Claude Code在文档生成中的协助
- 感谢计算机视觉社区的知识分享

## 📞 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/username/CV-Study-Notes/issues)
- **功能建议**: [GitHub Discussions](https://github.com/username/CV-Study-Notes/discussions)
- **邮件联系**: your.email@example.com

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个Star！**

Made with ❤️ by [yyq](https://github.com/yyq19990828) | Powered by [Claude Code](https://claude.ai/code)

</div>