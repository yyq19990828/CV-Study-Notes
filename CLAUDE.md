# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个计算机视觉学习笔记仓库，专门研究目标检测算法。每个子文件夹包含一种算法技术的官方实现代码和详细的中文技术分析文档。

## 项目结构

```
CV-Study-Notes/
├── DINO[IDEA-端到端检测]/        # DINO检测算法
│   ├── DINO/                    # 官方代码实现
│   └── [2022]DINO.md            # 技术分析文档
├── DINOv1/                       # DINO自监督学习
│   ├── dino/                    # 官方代码实现
│   └── [2021]DINO_自监督学习.md  # 技术分析文档
├── DINOv2/                       # DINOv2改进版
│   ├── dinov2/                  # 官方代码实现
│   └── [2024]DINOv2.md          # 技术分析文档
├── FCOS/                         # FCOS全卷积单阶段检测器
│   ├── FCOS/                    # 官方代码实现
│   └── [2019]FCOS技术报告.md    # 技术分析文档
└── YOLOX/                        # YOLOX无锚框检测器
    ├── YOLOX/                    # 官方代码实现
    └── 技术文档（待添加）
```

## 常用命令

### FCOS项目
```bash
# 训练FCOS模型
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x

# 测试FCOS模型
python tools/test_net.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
    MODEL.WEIGHT FCOS_imprv_R_50_FPN_1x.pth \
    TEST.IMS_PER_BATCH 4

# 运行演示
python demo/fcos_demo.py
```

### YOLOX项目
```bash
# 训练YOLOX模型
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache]

# 评估YOLOX模型
python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]

# 运行演示
python tools/demo.py image -n yolox-s -c /path/to/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result
```

## 代码架构说明

### FCOS架构
- **核心模型实现**: `fcos_core/modeling/rpn/fcos/` - FCOS检测头实现
- **损失函数**: `fcos_core/modeling/rpn/fcos/loss.py` - Focal Loss和IoU Loss
- **推理逻辑**: `fcos_core/modeling/rpn/fcos/inference.py` - 后处理和NMS
- **配置系统**: `configs/fcos/` - YAML配置文件

### YOLOX架构
- **核心模型**: `yolox/models/` - YOLOX网络结构
- **数据增强**: `yolox/data/data_augment.py` - Mosaic和MixUp
- **训练器**: `yolox/core/trainer.py` - 训练循环逻辑
- **实验配置**: `exps/default/` - 不同规模模型配置

## 工作流程

### 分析新论文时的标准流程

1. **阅读论文并提取核心技术点**
   - 识别主要创新点
   - 理解技术原理和动机
   - 记录关键公式和算法流程

2. **查找对应代码实现**
   - 使用Grep搜索关键函数名和变量名
   - 定位核心算法实现位置
   - 分析代码与论文的对应关系

3. **撰写技术分析文档**
   - 创建或更新算法文件夹下的.md文档
   - 包含以下内容：
     - 问题背景
     - 方法介绍（含公式）
     - 代码实现片段
     - 实验结果
     - 优势总结

4. **代码片段标注**
   - 引用代码时注明文件路径和行号
   - 示例：`fcos_core/modeling/rpn/fcos/fcos.py:123`
   - 保持代码片段简洁，突出核心逻辑

## 注意事项

- 所有技术文档使用中文撰写
- 保持文档结构清晰，使用适当的标题层级
- 公式使用LaTeX格式
- 代码片段需包含关键上下文
- 测试后清理临时文件