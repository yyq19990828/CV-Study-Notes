# FCOS 流程图集合

本文档包含FCOS算法的各种Mermaid流程图，展示算法的整体架构、数据流程、训练过程等关键环节。

## 1. 整体架构流程图

```mermaid
graph TB
    Input["输入图像<br/>[3, H, W]"]
    
    Input --> Backbone["骨干网络<br/>ResNet/ResNeXt"]
    Backbone --> FPN["特征金字塔FPN<br/>生成多尺度特征"]
    
    FPN --> P3["P3特征图<br/>[256, H/8, W/8]<br/>检测小物体(-1, 64)"]
    FPN --> P4["P4特征图<br/>[256, H/16, W/16]<br/>检测中小物体(64, 128)"]
    FPN --> P5["P5特征图<br/>[256, H/32, W/32]<br/>检测中等物体(128, 256)"]
    FPN --> P6["P6特征图<br/>[256, H/64, W/64]<br/>检测中大物体(256, 512)"]
    FPN --> P7["P7特征图<br/>[256, H/128, W/128]<br/>检测大物体(512, ∞)"]
    
    P3 --> Head1["FCOS检测头"]
    P4 --> Head2["FCOS检测头"]
    P5 --> Head3["FCOS检测头"]
    P6 --> Head4["FCOS检测头"]
    P7 --> Head5["FCOS检测头"]
    
    Head1 --> Pred["预测结果汇总"]
    Head2 --> Pred
    Head3 --> Pred
    Head4 --> Pred
    Head5 --> Pred
    
    Pred --> NMS["NMS后处理"]
    NMS --> Output["检测结果<br/>边界框+类别+得分"]
```

## 2. FCOS检测头详细结构

```mermaid
graph TB
    Feature["输入特征图<br/>[256, H, W]"]
    
    Feature --> ClsTower["分类塔<br/>4×(Conv3×3+GN+ReLU)"]
    Feature --> RegTower["回归塔<br/>4×(Conv3×3+GN+ReLU)"]
    
    ClsTower --> ClsConv["分类预测<br/>Conv3×3"]
    RegTower --> RegConv["回归预测<br/>Conv3×3"]
    RegTower --> CenterConv["中心度预测<br/>Conv3×3"]
    
    ClsConv --> ClsOut["分类输出<br/>[C, H, W]<br/>C=类别数"]
    RegConv --> Scale["尺度变换<br/>scale[l]"]
    Scale --> RegOut["回归输出<br/>[4, H, W]<br/>(l,t,r,b)"]
    CenterConv --> CenterOut["中心度输出<br/>[1, H, W]"]
    
    style ClsTower fill:#ffd700
    style RegTower fill:#90ee90
```

## 3. 训练流程图

```mermaid
graph LR
    Start["训练开始"] --> LoadData["加载训练数据"]
    LoadData --> Forward["前向传播"]
    
    Forward --> GenTargets["生成训练目标"]
    GenTargets --> AssignSamples["样本分配<br/>1.尺度分配<br/>2.中心采样"]
    
    AssignSamples --> ComputeLoss["计算损失"]
    
    ComputeLoss --> ClsLoss["分类损失<br/>Focal Loss"]
    ComputeLoss --> RegLoss["回归损失<br/>IoU/GIoU Loss"]
    ComputeLoss --> CenterLoss["中心度损失<br/>BCE Loss"]
    
    ClsLoss --> TotalLoss["总损失"]
    RegLoss --> TotalLoss
    CenterLoss --> TotalLoss
    
    TotalLoss --> Backward["反向传播"]
    Backward --> UpdateParams["更新参数"]
    UpdateParams --> CheckIter{迭代完成?}
    
    CheckIter -->|否| Forward
    CheckIter -->|是| End["训练结束"]
```

## 4. 正负样本分配流程

```mermaid
graph TB
    Start["特征图位置<br/>(x, y)"]
    
    Start --> MapToImage["映射到原图<br/>(xs, ys) = (x*s+s/2, y*s+s/2)"]
    MapToImage --> CheckInBox{位置是否在<br/>任意GT框内?}
    
    CheckInBox -->|否| Negative["负样本"]
    CheckInBox -->|是| CheckCenter{是否在GT<br/>中心区域?}
    
    CheckCenter -->|否| Negative
    CheckCenter -->|是| CheckScale{目标尺寸是否<br/>匹配当前层级?}
    
    CheckScale -->|否| Negative
    CheckScale -->|是| CheckMulti{是否落在<br/>多个GT内?}
    
    CheckMulti -->|否| Positive["正样本"]
    CheckMulti -->|是| SelectMin["选择面积<br/>最小的GT"]
    SelectMin --> Positive
    
    Positive --> ComputeTargets["计算回归目标<br/>l*, t*, r*, b*"]
    ComputeTargets --> ComputeCenterness["计算中心度<br/>centerness*"]
```

## 5. 中心度计算流程

```mermaid
graph TB
    Input["回归目标<br/>(l*, t*, r*, b*)"]
    
    Input --> LR["计算左右比值<br/>lr_ratio = min(l*, r*) / max(l*, r*)"]
    Input --> TB["计算上下比值<br/>tb_ratio = min(t*, b*) / max(t*, b*)"]
    
    LR --> Multiply["相乘<br/>centerness = lr_ratio × tb_ratio"]
    TB --> Multiply
    
    Multiply --> Sqrt["开方<br/>centerness = sqrt(centerness)"]
    Sqrt --> Output["中心度值<br/>范围: [0, 1]"]
    
    Output --> Usage1["训练时：作为回归损失权重"]
    Output --> Usage2["推理时：乘以分类得分"]
```

## 6. 推理流程图

```mermaid
graph TB
    Image["输入图像"]
    
    Image --> Extract["特征提取<br/>Backbone + FPN"]
    Extract --> MultiScale["多尺度预测<br/>P3, P4, P5, P6, P7"]
    
    MultiScale --> Decode["逐层解码"]
    
    Decode --> DecodeBox["解码边界框<br/>x1 = xs - l<br/>y1 = ys - t<br/>x2 = xs + r<br/>y2 = ys + b"]
    Decode --> ScoreFusion["得分融合<br/>score = cls_score × centerness"]
    
    DecodeBox --> Filter["阈值过滤<br/>score > threshold"]
    ScoreFusion --> Filter
    
    Filter --> TopK["Top-K选择<br/>每层最多1000个"]
    TopK --> Merge["合并所有层级预测"]
    Merge --> NMS["NMS去重<br/>IoU阈值: 0.6"]
    NMS --> Final["最终检测结果"]
```

## 7. 损失计算详细流程

```mermaid
graph TB
    Predictions["模型预测<br/>cls, reg, centerness"]
    Targets["真实标签<br/>GT boxes, labels"]
    
    Predictions --> SplitPos["区分正负样本"]
    Targets --> SplitPos
    
    SplitPos --> PosSamples["正样本"]
    SplitPos --> NegSamples["负样本"]
    
    PosSamples --> CalcCenterTarget["计算中心度目标"]
    
    CalcCenterTarget --> RegLoss["回归损失<br/>IoU Loss × centerness_target"]
    CalcCenterTarget --> CenterLoss["中心度损失<br/>BCE(pred, target)"]
    
    PosSamples --> ClsLossPos["正样本分类损失"]
    NegSamples --> ClsLossNeg["负样本分类损失"]
    
    ClsLossPos --> FocalLoss["Focal Loss<br/>α=0.25, γ=2.0"]
    ClsLossNeg --> FocalLoss
    
    FocalLoss --> NormByPos["除以正样本数"]
    RegLoss --> NormByCenterness["除以中心度之和"]
    CenterLoss --> NormByPos2["除以正样本数"]
    
    NormByPos --> TotalLoss["总损失<br/>L = L_cls + L_reg + L_centerness"]
    NormByCenterness --> TotalLoss
    NormByPos2 --> TotalLoss
```

## 8. 特征金字塔数据流

```mermaid
graph LR
    C3["C3<br/>[512, H/8, W/8]"] --> Lateral3["1×1 Conv<br/>降维到256"]
    C4["C4<br/>[1024, H/16, W/16]"] --> Lateral4["1×1 Conv<br/>降维到256"]
    C5["C5<br/>[2048, H/32, W/32]"] --> Lateral5["1×1 Conv<br/>降维到256"]
    
    Lateral5 --> P5["P5<br/>[256, H/32, W/32]"]
    
    P5 --> Upsample5["2×上采样"]
    Upsample5 --> Add4["逐元素相加"]
    Lateral4 --> Add4
    Add4 --> Smooth4["3×3 Conv<br/>平滑"]
    Smooth4 --> P4["P4<br/>[256, H/16, W/16]"]
    
    P4 --> Upsample4["2×上采样"]
    Upsample4 --> Add3["逐元素相加"]
    Lateral3 --> Add3
    Add3 --> Smooth3["3×3 Conv<br/>平滑"]
    Smooth3 --> P3["P3<br/>[256, H/8, W/8]"]
    
    P5 --> Downsample6["stride=2 Conv"]
    Downsample6 --> P6["P6<br/>[256, H/64, W/64]"]
    
    P6 --> Downsample7["stride=2 Conv"]
    Downsample7 --> P7["P7<br/>[256, H/128, W/128]"]
```

## 9. 维度变化示意图

```mermaid
graph TB
    Input["输入: [B, 3, 800, 1333]"]
    
    Input --> Backbone["ResNet-50"]
    Backbone --> C3C4C5["C3: [B, 512, 100, 167]<br/>C4: [B, 1024, 50, 84]<br/>C5: [B, 2048, 25, 42]"]
    
    C3C4C5 --> FPN["FPN处理"]
    
    FPN --> Features["P3: [B, 256, 100, 167]<br/>P4: [B, 256, 50, 84]<br/>P5: [B, 256, 25, 42]<br/>P6: [B, 256, 13, 21]<br/>P7: [B, 256, 7, 11]"]
    
    Features --> Head["FCOS Head"]
    
    Head --> Outputs["每层输出:<br/>分类: [B, 80, Hi, Wi]<br/>回归: [B, 4, Hi, Wi]<br/>中心度: [B, 1, Hi, Wi]"]
    
    Outputs --> Reshape["重塑为列表"]
    Reshape --> Final["分类: 5×[B, Hi×Wi, 80]<br/>回归: 5×[B, Hi×Wi, 4]<br/>中心度: 5×[B, Hi×Wi, 1]"]
```

## 10. 改进技术对比流程

```mermaid
graph TB
    BaselineFCOS["基础FCOS<br/>AP: 36.6%"]
    
    BaselineFCOS --> AddCenterness["添加中心度机制<br/>AP: 37.6% (提升1.0%)"]
    AddCenterness --> AddCenterSample["添加中心采样<br/>AP: 38.4% (提升0.8%)"]
    AddCenterSample --> AddGIoU["添加GIoU Loss<br/>AP: 39.0% (提升0.6%)"]
    AddGIoU --> AddNorm["添加归一化回归<br/>AP: 39.5% (提升0.5%)"]
    
    AddNorm --> ImprovedFCOS["改进版FCOS<br/>总AP: 39.5%"]
    
    ImprovedFCOS --> AddDCN["添加可变形卷积<br/>AP: 42.3%"]
    AddDCN --> LargerBackbone["更大骨干网络<br/>ResNeXt-101-64x4d"]
    LargerBackbone --> MultiScaleTest["多尺度测试<br/>AP: 49.0%"]
    
    style BaselineFCOS fill:#ffcccc
    style ImprovedFCOS fill:#ccffcc
    style MultiScaleTest fill:#ccccff
```

## 总结

这些流程图全面展示了FCOS算法的各个关键环节：

1. **整体架构**：展示了从输入到输出的完整流程
2. **检测头结构**：详细说明了分类、回归和中心度三个分支
3. **训练流程**：包含了前向传播、损失计算和反向传播
4. **样本分配**：说明了如何确定正负样本
5. **中心度机制**：展示了中心度的计算和使用方式
6. **推理流程**：从特征提取到最终检测结果
7. **损失计算**：三种损失的详细计算过程
8. **FPN结构**：特征金字塔的构建过程
9. **维度变化**：数据在网络中的维度变化
10. **改进对比**：各项改进技术的贡献

这些流程图有助于深入理解FCOS的工作原理和实现细节。