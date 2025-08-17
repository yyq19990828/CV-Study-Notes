# YOLOv6算法流程图集合

## 1. YOLOv6整体架构流程图

```mermaid
graph TB
    Input["输入图像<br/>[B, 3, H, W]"] --> Backbone["EfficientRep骨干网络<br/>RepBlock/CSPStackRep"]
    
    Backbone --> C3["C3特征<br/>[B, C3, H/8, W/8]"]
    Backbone --> C4["C4特征<br/>[B, C4, H/16, W/16]"]
    Backbone --> C5["C5特征<br/>[B, C5, H/32, W/32]"]
    
    C3 --> Neck["Rep-PAN颈部网络"]
    C4 --> Neck
    C5 --> Neck
    
    Neck --> P3["P3特征<br/>[B, 256, H/8, W/8]"]
    Neck --> P4["P4特征<br/>[B, 256, H/16, W/16]"]
    Neck --> P5["P5特征<br/>[B, 256, H/32, W/32]"]
    
    P3 --> Head3["高效解耦头"]
    P4 --> Head4["高效解耦头"]
    P5 --> Head5["高效解耦头"]
    
    Head3 --> Pred3["预测输出<br/>分类+回归"]
    Head4 --> Pred4["预测输出<br/>分类+回归"]
    Head5 --> Pred5["预测输出<br/>分类+回归"]
    
    Pred3 --> NMS["NMS后处理"]
    Pred4 --> NMS
    Pred5 --> NMS
    
    NMS --> Output["最终检测结果"]
    
    style Backbone fill:#e1f5fe
    style Neck fill:#fff3e0
    style Head3 fill:#e8f5e8
    style Head4 fill:#e8f5e8
    style Head5 fill:#e8f5e8
```

## 2. EfficientRep骨干网络架构图

```mermaid
graph TB
    subgraph SmallModels["小模型 (N/T/S)"]
        Input_S["输入"] --> RepBlock1["RepBlock<br/>stride=2"]
        RepBlock1 --> RepBlock2["RepBlock<br/>stride=2"]
        RepBlock2 --> RepBlock3["RepBlock<br/>stride=2"]
        RepBlock3 --> RepBlock4["RepBlock<br/>stride=2"]
        RepBlock4 --> RepBlock5["RepBlock<br/>stride=2"]
    end
    
    subgraph LargeModels["大模型 (M/L)"]
        Input_L["输入"] --> CSP1["CSPStackRep<br/>stride=2"]
        CSP1 --> CSP2["CSPStackRep<br/>stride=2"]
        CSP2 --> CSP3["CSPStackRep<br/>stride=2"]
        CSP3 --> CSP4["CSPStackRep<br/>stride=2"]
        CSP4 --> CSP5["CSPStackRep<br/>stride=2"]
    end
    
    style RepBlock1 fill:#e8f5e8
    style CSP1 fill:#fff3e0
```

## 3. RepBlock重参数化流程图

```mermaid
graph LR
    subgraph Training["训练阶段"]
        X_train["输入X"] --> Conv3x3["3×3卷积"]
        X_train --> Conv1x1["1×1卷积"]
        X_train --> Identity["恒等映射<br/>(如果通道数相同)"]
        
        Conv3x3 --> Add["相加"]
        Conv1x1 --> Add
        Identity --> Add
        Add --> Y_train["输出Y"]
    end
    
    subgraph Inference["推理阶段"]
        X_infer["输入X"] --> MergedConv["融合后的<br/>3×3卷积"]
        MergedConv --> Y_infer["输出Y"]
    end
    
    Training -->|"参数融合"| Inference
    
    style Training fill:#e1f5fe
    style Inference fill:#e8f5e8
```

## 4. Rep-PAN with BiC颈部网络流程图 (v3.0)

```mermaid
graph TB
    subgraph RepPAN["Rep-PAN颈部网络"]
        C3["C3 (低层)"] --> P5_1["第一阶段融合"]
        C4["C4 (中层)"] --> P5_1
        C5["C5 (高层)"] --> P5_1
        
        P5_1 --> Up1["上采样"]
        Up1 --> P4_1["第二阶段融合"]
        C4 --> P4_1
        
        P4_1 --> Up2["上采样"]
        Up2 --> P3_1["第三阶段融合"]
        C3 --> P3_1
        
        subgraph BiC["BiC模块 (v3.0新增)"]
            P3_1 --> BiC_3["BiC融合<br/>P3 = Cat(Up(P4), C3, C2)"]
            P4_1 --> BiC_4["BiC融合<br/>P4 = Cat(Up(P5), C4, C3)"]
        end
        
        P3_1 --> Down1["下采样"]
        Down1 --> P4_2["第四阶段融合"]
        P4_1 --> P4_2
        
        P4_2 --> Down2["下采样"]
        Down2 --> P5_2["第五阶段融合"]
        P5_1 --> P5_2
    end
    
    P3_1 --> P3_out["P3输出"]
    P4_2 --> P4_out["P4输出"]
    P5_2 --> P5_out["P5输出"]
    
    style BiC fill:#fce4ec
```

## 5. 高效解耦头结构图

```mermaid
graph TB
    Feature["输入特征<br/>[B, 256, H, W]"] --> Stem["1×1 Conv<br/>降维"]
    
    Stem --> ClsBranch["分类分支"]
    Stem --> RegBranch["回归分支"]
    
    subgraph Classification["分类分支"]
        ClsBranch --> ClsConv["3×3 Conv"]
        ClsConv --> ClsOut["分类输出<br/>[B, num_classes, H, W]"]
    end
    
    subgraph Regression["回归分支"]
        RegBranch --> RegConv["3×3 Conv"]
        RegConv --> RegOut["回归输出<br/>[B, 4, H, W]"]
    end
    
    style Classification fill:#e1f5fe
    style Regression fill:#fff3e0
```

## 6. TAL标签分配流程图

```mermaid
graph TB
    Start["开始标签分配"] --> CalcMetric["计算对齐度量<br/>t = s^α × u^β"]
    
    CalcMetric --> TopK["选择Top-k个anchor<br/>(默认k=13)"]
    
    TopK --> CenterCheck{"anchor中心<br/>是否在GT内?"}
    
    CenterCheck -->|是| Positive["正样本"]
    CenterCheck -->|否| Negative["负样本"]
    
    Positive --> Normalize["归一化对齐度量<br/>t̂ = normalize(t)"]
    
    Normalize --> AssignLabel["分配软标签"]
    
    AssignLabel --> Loss["计算损失"]
    
    style CalcMetric fill:#fff3e0
    style Normalize fill:#e8f5e8
```

## 7. AAT锚点辅助训练流程图 (v3.0)

```mermaid
graph TB
    subgraph TrainingPhase["训练阶段"]
        Features["特征图"] --> MainHead["主检测头<br/>(Anchor-free)"]
        Features --> AuxHead["辅助检测头<br/>(Anchor-based)"]
        
        MainHead --> MainLoss["主损失"]
        AuxHead --> AuxLoss["辅助损失"]
        
        MainLoss --> TotalLoss["总损失"]
        AuxLoss --> TotalLoss
        
        TotalLoss --> Backprop["反向传播"]
    end
    
    subgraph InferencePhase["推理阶段"]
        Features_Inf["特征图"] --> MainHead_Inf["主检测头<br/>(Anchor-free)"]
        MainHead_Inf --> Output["检测输出"]
        AuxHead_Removed["辅助头已移除<br/>❌"]
    end
    
    style TrainingPhase fill:#e1f5fe
    style InferencePhase fill:#e8f5e8
    style AuxHead_Removed fill:#ffebee
```

## 8. 自蒸馏训练流程图

```mermaid
graph TB
    subgraph TeacherModel["教师模型"]
        Input_T["输入"] --> Teacher["预训练模型"]
        Teacher --> T_Cls["分类预测"]
        Teacher --> T_Reg["回归预测"]
    end
    
    subgraph StudentModel["学生模型"]
        Input_S["输入"] --> Student["训练中模型"]
        Student --> S_Cls["分类预测"]
        Student --> S_Reg["回归预测"]
    end
    
    T_Cls --> KD_Cls["KL散度<br/>分类蒸馏"]
    S_Cls --> KD_Cls
    
    T_Reg --> KD_Reg["KL散度<br/>回归蒸馏"]
    S_Reg --> KD_Reg
    
    GT["真实标签"] --> Hard_Loss["硬标签损失"]
    S_Cls --> Hard_Loss
    S_Reg --> Hard_Loss
    
    KD_Cls --> Total["总损失<br/>L = L_hard + α·L_KD"]
    KD_Reg --> Total
    Hard_Loss --> Total
    
    Total --> Update["更新学生模型"]
    
    style TeacherModel fill:#fff3e0
    style StudentModel fill:#e1f5fe
```

## 9. DLD解耦定位蒸馏流程图 (v3.0小模型专用)

```mermaid
graph LR
    subgraph Training["训练阶段"]
        Feat_Train["特征"] --> Light["轻量回归头"]
        Feat_Train --> Heavy["增强回归头<br/>(含DFL)"]
        
        Light --> Light_Pred["轻量预测"]
        Heavy --> Heavy_Pred["增强预测"]
        
        Heavy_Pred --> Distill["蒸馏损失"]
        Light_Pred --> Distill
        
        GT_Train["GT"] --> Heavy_Loss["增强头损失"]
        Heavy_Pred --> Heavy_Loss
    end
    
    subgraph Inference["推理阶段"]
        Feat_Inf["特征"] --> Light_Inf["轻量回归头"]
        Light_Inf --> Final_Pred["最终预测"]
        
        Heavy_Removed["增强头已移除<br/>❌"]
    end
    
    style Training fill:#e1f5fe
    style Inference fill:#e8f5e8
    style Heavy_Removed fill:#ffebee
```

## 10. YOLOv6-N6/S6/M6/L6扩展架构流程图

```mermaid
graph TB
    Input["输入图像<br/>[B, 3, 1280, 1280]"] --> Backbone["6阶段骨干网络"]
    
    Backbone --> C3["C3: stride=8"]
    Backbone --> C4["C4: stride=16"]
    Backbone --> C5["C5: stride=32"]
    Backbone --> C6["C6: stride=64<br/>(新增阶段)"]
    
    C3 --> Neck["扩展Rep-PAN"]
    C4 --> Neck
    C5 --> Neck
    C6 --> Neck
    
    Neck --> P3["P3: 小目标"]
    Neck --> P4["P4: 中目标"]
    Neck --> P5["P5: 大目标"]
    Neck --> P6["P6: 超大目标<br/>(新增)"]
    
    P3 --> Det["多尺度检测"]
    P4 --> Det
    P5 --> Det
    P6 --> Det
    
    Det --> Output["高分辨率检测结果"]
    
    style C6 fill:#fce4ec
    style P6 fill:#fce4ec
```

## 11. 量化部署优化流程图

```mermaid
graph TB
    Model["YOLOv6模型"] --> RepOpt["RepOptimizer<br/>梯度重参数化"]
    
    RepOpt --> PTQ["PTQ量化<br/>(后训练量化)"]
    RepOpt --> QAT["QAT量化<br/>(量化感知训练)"]
    
    PTQ --> Deploy_PTQ["部署模型<br/>43.3% AP @ 869 FPS"]
    
    QAT --> CWDistill["通道级蒸馏"]
    CWDistill --> GraphOpt["图优化"]
    GraphOpt --> Deploy_QAT["优化部署模型"]
    
    Deploy_PTQ --> TensorRT["TensorRT推理"]
    Deploy_QAT --> TensorRT
    
    style RepOpt fill:#fff3e0
    style Deploy_PTQ fill:#e8f5e8
```

## 12. 训练策略演进流程图

```mermaid
graph LR
    subgraph V1["YOLOv6原版 (2022)"]
        TAL1["TAL标签分配"]
        SD1["自蒸馏"]
        Epochs1["300 epochs"]
    end
    
    subgraph V3["YOLOv6 v3.0 (2023)"]
        TAL3["TAL标签分配"]
        BiC3["BiC模块"]
        AAT3["AAT辅助训练"]
        DLD3["DLD蒸馏"]
        Epochs3["400 epochs"]
    end
    
    V1 -->|"升级"| V3
    
    subgraph Improvements["性能提升"]
        AP_Boost["AP: +1.5~2%"]
        Speed_Maintain["速度: 保持/提升"]
    end
    
    V3 --> Improvements
    
    style V1 fill:#f0f0f0
    style V3 fill:#e1f5fe
    style Improvements fill:#e8f5e8
```

---

**说明**：
- 所有流程图基于YOLOv6原版论文（2022）和v3.0论文（2023）的设计
- 标注了v3.0的新增特性（BiC、AAT、DLD等）
- 维度标注基于COCO数据集的标准设置
- 颜色编码用于区分不同组件类型