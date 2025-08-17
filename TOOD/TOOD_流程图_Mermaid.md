# TOOD算法流程图集合

## 1. TOOD整体架构流程图

```mermaid
graph TB
    Input["输入图像<br/>[B, 3, H, W]"] --> Backbone["骨干网络<br/>ResNet + FPN"]
    Backbone --> FPN_Features["FPN特征<br/>[B, 256, H/8, W/8] to <br/>[B, 256, H/128, W/128]"]
    
    FPN_Features --> T_Head["Task-aligned Head (T-Head)"]
    T_Head --> Predictions["预测结果<br/>分类 + 回归"]

    Predictions --> TAL["Task Alignment Learning <br/> (TAL)"]
    TAL --> Alignment_Metric["对齐度量计算<br/>t = s^α × u^β"]
    Alignment_Metric --> Sample_Assignment["样本分配"]
    Sample_Assignment --> Loss_Computation["损失计算"]
    
    Loss_Computation --> Backprop["反向传播"]
    Backprop --> T_Head
    
    style T_Head fill:#e1f5fe
    style TAL fill:#f3e5f5
    style Alignment_Metric fill:#fff3e0
```

## 2. Task-aligned Head (T-Head) 详细流程图

```mermaid
graph TB
    subgraph THeadArch["T-Head Architecture"]
        FPN_In["FPN特征输入<br/>[B, 256, H, W]"] --> Conv_Stack["堆叠卷积层<br/>6层 Conv + ReLU"]
        
        Conv_Stack --> Inter_Features["任务交互特征<br/>X_inter_1 to X_inter_6"]
        Inter_Features --> Feature_Concat["特征连接<br/>[B, 256×6, H, W]"]
        
        Feature_Concat --> Task_Decomp_Cls["分类任务分解<br/>TaskDecomposition"]
        Feature_Concat --> Task_Decomp_Reg["回归任务分解<br/>TaskDecomposition"]
        
        subgraph TaskDecompModule["TaskDecomposition Module"]
            GAP["全局平均池化<br/>GAP"] --> FC1["FC层 256×6→256×6//8"]
            FC1 --> ReLU["ReLU激活"]
            ReLU --> FC2["FC层 256×6//8→6"]
            FC2 --> Sigmoid["Sigmoid激活"]
            Sigmoid --> Layer_Weights["层注意力权重<br/>w ∈ R^6"]
        end
        
        Task_Decomp_Cls --> Cls_Features["分类特征<br/>[B, 256, H, W]"]
        Task_Decomp_Reg --> Reg_Features["回归特征<br/>[B, 256, H, W]"]
        
        Cls_Features --> Cls_Pred["分类预测<br/>[B, 80, H, W]"]
        Reg_Features --> Reg_Pred["回归预测<br/>[B, 4, H, W]"]
        
        Feature_Concat --> Prob_Map["概率图生成<br/>M ∈ R^(H×W×1)"]
        Feature_Concat --> Offset_Map["偏移图生成<br/>O ∈ R^(H×W×8)"]
        
        Cls_Pred --> Alignment_Cls["对齐分类预测<br/>P_align = √(P × M)"]
        Reg_Pred --> Alignment_Reg["对齐回归预测<br/>B_align(i,j,c) = B(i+O(i,j,2×c), j+O(i,j,2×c+1), c)"]
    end
    
    style Task_Decomp_Cls fill:#e8f5e8
    style Task_Decomp_Reg fill:#fff3e0
    style Alignment_Cls fill:#e1f5fe
    style Alignment_Reg fill:#fce4ec
```

## 3. Task Alignment Learning (TAL) 流程图

```mermaid
graph TB
    subgraph TALStrategy["TAL Training Strategy"]
        Pred_Cls["分类预测<br/>scores [N, 80]"] --> Score_Extract["提取对应类别得分<br/>s = scores[:, gt_labels]"]
        Pred_Reg["回归预测<br/>decode_bboxes [N, 4]"] --> IoU_Calc["计算IoU<br/>u = IoU(pred_boxes, gt_boxes)"]
        
        Score_Extract --> Alignment_Metric["对齐度量计算<br/>t = s^α × u^β"]
        IoU_Calc --> Alignment_Metric
        
        Alignment_Metric --> TopK_Selection["Top-k样本选择<br/>选择前m个最大t值作为正样本"]
        TopK_Selection --> Center_Constraint["中心约束<br/>锚点必须在GT内部"]
        
        Center_Constraint --> Positive_Samples["正样本集合<br/>Positive anchors"]
        Center_Constraint --> Negative_Samples["负样本集合<br/>Negative anchors"]
        
        Positive_Samples --> Normalization["对齐度量归一化<br/>t̂ = normalize(t)"]
        Normalization --> Task_Aligned_Loss["任务对齐损失"]
        
        subgraph TaskLossFuncs["Task-aligned Loss Functions"]
            Task_Aligned_Loss --> Cls_Loss["分类损失<br/>L_cls = Σ|t̂ᵢ - sᵢ|^γ × BCE(sᵢ, t̂ᵢ)"]
            Task_Aligned_Loss --> Reg_Loss["回归损失<br/>L_reg = Σt̂ᵢ × L_GIoU(bᵢ, b̄ᵢ)"]
        end
        
        Cls_Loss --> Total_Loss["总损失<br/>L = L_cls + L_reg"]
        Reg_Loss --> Total_Loss
    end
    
    style Alignment_Metric fill:#fff3e0
    style Task_Aligned_Loss fill:#f3e5f5
    style Total_Loss fill:#ffebee
```

## 4. 训练阶段完整流程图

```mermaid
graph TB
    subgraph TrainingPipeline["Training Pipeline"]
        Start["开始训练"] --> Epoch_Check{"epoch < initial_epoch?<br/>(前4个epoch)"}
        
        Epoch_Check -->|Yes| Initial_Stage["初始训练阶段"]
        Epoch_Check -->|No| Alignment_Stage["对齐训练阶段"]
        
        subgraph InitialStage["Initial Training Stage"]
            Initial_Stage --> ATSS_Assigner["ATSS分配器<br/>基于IoU的样本分配"]
            ATSS_Assigner --> Focal_Loss["Focal Loss<br/>标准分类损失"]
        end
        
        subgraph AlignmentStage["Alignment Training Stage"]
            Alignment_Stage --> Forward_Pass["前向传播<br/>T-Head预测"]
            Forward_Pass --> Alignment_Computation["对齐度量计算<br/>t = s^α × u^β"]
            Alignment_Computation --> TAL_Assignment["TAL样本分配<br/>Top-k选择"]
            TAL_Assignment --> TAL_Loss["TAL损失计算"]
        end
        
        Focal_Loss --> Backward["反向传播"]
        TAL_Loss --> Backward
        Backward --> Update["参数更新"]
        Update --> Next_Iter{"下一次迭代"}
        
        Next_Iter --> Epoch_Check
    end
    
    style Initial_Stage fill:#e8f5e8
    style Alignment_Stage fill:#e1f5fe
    style TAL_Loss fill:#f3e5f5
```

## 5. 推理阶段流程图

```mermaid
graph LR
    subgraph InferencePipeline["Inference Pipeline"]
        Input_Img["输入图像"] --> Resize["图像缩放<br/>短边800像素"]
        Resize --> Network["网络前向传播<br/>T-Head + TAL训练的权重"]
        
        Network --> Raw_Pred["原始预测<br/>分类得分 + 边界框"]
        Raw_Pred --> Confidence_Filter["置信度过滤<br/>threshold = 0.05"]
        
        Confidence_Filter --> Top_K["Top-1000选择<br/>每个FPN层级"]
        Top_K --> NMS["非极大值抑制<br/>IoU threshold = 0.6"]
        
        NMS --> Final_Results["最终检测结果<br/>Top-100 per image"]
    end
    
    style Network fill:#e1f5fe
    style Final_Results fill:#e8f5e8
```

## 6. 任务对齐机制详细流程

```mermaid
graph TB
    subgraph TaskAlignMech["Task Alignment Mechanism"]
        Cls_Branch["分类分支预测<br/>P ∈ R^(H×W×80)"]
        Reg_Branch["回归分支预测<br/>B ∈ R^(H×W×4)"]
        Interactive_Features["交互特征<br/>X_inter"]
        
        Interactive_Features --> Prob_Gen["概率图生成<br/>M = σ(conv(X_inter))"]
        Interactive_Features --> Offset_Gen["偏移图生成<br/>O = conv(X_inter)"]
        
        Cls_Branch --> Spatial_Align_Cls["空间对齐分类<br/>P_align = √(P × M)"]
        Reg_Branch --> Spatial_Align_Reg["空间对齐回归<br/>通过双线性插值应用偏移"]
        
        Prob_Gen --> Spatial_Align_Cls
        Offset_Gen --> Spatial_Align_Reg
        
        Spatial_Align_Cls --> Aligned_Output["对齐输出"]
        Spatial_Align_Reg --> Aligned_Output
        
        Aligned_Output --> Benefits["任务对齐优势"]
        
        subgraph AlignBenefits["Alignment Benefits"]
            Benefits --> Benefit1["1.最优锚点对齐<br/>分类和回归使用相同锚点"]
            Benefits --> Benefit2["2.预测质量提升<br/>减少空间错位"]
            Benefits --> Benefit3["3.NMS友好<br/>减少冗余和错误框"]
        end
    end
    
    style Spatial_Align_Cls fill:#e8f5e8
    style Spatial_Align_Reg fill:#fff3e0
    style Aligned_Output fill:#e1f5fe
```

## 7. 损失函数计算流程

```mermaid
graph TB
    subgraph LossCompFlow["Loss Computation Flow"]
        Predictions["网络预测<br/>scores, bboxes"] --> GT_Matching["与GT匹配"]
        GT_Info["GT信息<br/>gt_bboxes, gt_labels"] --> GT_Matching
        
        GT_Matching --> Alignment_Score["对齐得分计算<br/>t = s^α × u^β"]
        
        Alignment_Score --> Sample_Selection["样本选择<br/>Top-m正样本"]
        Sample_Selection --> Norm_Alignment["归一化对齐得分<br/>t̂ = normalize(t)"]
        
        Norm_Alignment --> Cls_Loss_Comp["分类损失计算"]
        Norm_Alignment --> Reg_Loss_Comp["回归损失计算"]
        
        subgraph ClassificationLoss["Classification Loss"]
            Cls_Loss_Comp --> Positive_Cls["正样本分类损失<br/>L_cls_pos = Σ|t̂ᵢ - sᵢ|^γ × BCE(sᵢ, t̂ᵢ)"]
            Cls_Loss_Comp --> Negative_Cls["负样本分类损失<br/>L_cls_neg = Σs_j^γ × BCE(s_j, 0)"]
        end
        
        subgraph RegressionLoss["Regression Loss"]
            Reg_Loss_Comp --> Weighted_Reg["加权回归损失<br/>L_reg = Σt̂ᵢ × L_GIoU(bᵢ, b̄ᵢ)"]
        end
        
        Positive_Cls --> Final_Loss["最终损失<br/>L = L_cls_pos + L_cls_neg + L_reg"]
        Negative_Cls --> Final_Loss
        Weighted_Reg --> Final_Loss
    end
    
    style Alignment_Score fill:#fff3e0
    style Final_Loss fill:#ffebee
    style Positive_Cls fill:#e8f5e8
    style Weighted_Reg fill:#e1f5fe
```

## 8. 网络架构层次图

```mermaid
graph TB
    subgraph TOODNetArch["TOOD Network Architecture"]
        Input["输入图像<br/>[B, 3, H, W]"]
        
        subgraph BackboneNet["Backbone"]
            ResNet["ResNet-50/101<br/>特征提取"]
            FPN["Feature Pyramid Network<br/>多尺度特征融合"]
        end
        
        subgraph THeadNet["T-Head"]
            Conv_Tower["6层卷积塔<br/>任务交互特征提取"]
            
            subgraph TaskDecompNet["Task Decomposition"]
                Layer_Attention["层注意力机制<br/>动态特征选择"]
                Cls_Decomp["分类任务分解"]
                Reg_Decomp["回归任务分解"]
            end
            
            subgraph PredAlignNet["Prediction Alignment"]
                Prob_Map_Gen["概率图生成<br/>分类对齐"]
                Offset_Map_Gen["偏移图生成<br/>回归对齐"]
            end
            
            Cls_Head["分类预测头<br/>[B, 80, H, W]"]
            Reg_Head["回归预测头<br/>[B, 4, H, W]"]
        end
        
        Input --> ResNet
        ResNet --> FPN
        FPN --> Conv_Tower
        
        Conv_Tower --> Layer_Attention
        Layer_Attention --> Cls_Decomp
        Layer_Attention --> Reg_Decomp
        
        Cls_Decomp --> Cls_Head
        Reg_Decomp --> Reg_Head
        
        Conv_Tower --> Prob_Map_Gen
        Conv_Tower --> Offset_Map_Gen
        
        Cls_Head --> Aligned_Cls["对齐分类输出"]
        Reg_Head --> Aligned_Reg["对齐回归输出"]
        Prob_Map_Gen --> Aligned_Cls
        Offset_Map_Gen --> Aligned_Reg
        
        subgraph TrainingComponents["Training Components"]
            TAL_Assigner["TAL样本分配器"]
            TAL_Loss["TAL损失函数"]
        end
        
        Aligned_Cls --> TAL_Assigner
        Aligned_Reg --> TAL_Assigner
        TAL_Assigner --> TAL_Loss
    end
    
    style THeadNet fill:#e1f5fe
    style TaskDecompNet fill:#fff3e0
    style PredAlignNet fill:#e8f5e8
    style TrainingComponents fill:#f3e5f5
```

---

**说明**:
- 所有流程图基于TOOD论文的原始设计和MMDetection代码实现
- α和β是任务对齐度量的超参数，论文中推荐α=1, β=6
- m是正样本选择的数量，论文中使用m=13
- γ是focal loss的聚焦参数，设置为2
- 图中的维度标注基于COCO数据集(80个类别)的标准设置