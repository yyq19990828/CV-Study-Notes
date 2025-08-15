# DINOv2 数据流程图（Mermaid版）

> 注：如果某些图表无法正常显示，可以使用支持Mermaid的Markdown编辑器（如VSCode + Mermaid插件、Typora等）查看

## 一、整体架构流程图

```mermaid
graph TB
    Input["输入图像<br/>B×3×224×224"]
    
    Input --> Augment["数据增强<br/>多视角生成"]
    
    Augment --> Teacher_Aug["教师增强<br/>Global Crops"]
    Augment --> Student_Aug["学生增强<br/>Global + Local Crops"]
    
    Teacher_Aug --> Teacher_Backbone["教师网络<br/>ViT (冻结梯度)"]
    Student_Aug --> Student_Backbone["学生网络<br/>ViT (可训练)"]
    
    Teacher_Backbone --> Teacher_Features["教师特征<br/>CLS + Patch + Register"]
    Student_Backbone --> Student_Features["学生特征<br/>CLS + Patch + Register"]
    
    Teacher_Features --> Teacher_Head["教师投影头<br/>3层MLP"]
    Student_Features --> Student_Head["学生投影头<br/>3层MLP"]
    
    Teacher_Head --> Teacher_Output["教师输出<br/>B×65536"]
    Student_Head --> Student_Output["学生输出<br/>B×65536"]
    
    Teacher_Output --> Center["中心化处理<br/>动态center更新"]
    Center --> Teacher_Softmax["教师Softmax<br/>温度=0.04-0.07"]
    
    Student_Output --> Student_Logsoftmax["学生LogSoftmax<br/>温度=0.1"]
    
    Teacher_Softmax --> DINO_Loss["DINO损失<br/>全局特征对齐"]
    Student_Logsoftmax --> DINO_Loss
    
    %% iBOT分支
    Student_Features --> Student_Patch["学生Patch特征<br/>(遮挡)"]
    Teacher_Features --> Teacher_Patch["教师Patch特征<br/>(完整)"]
    
    Student_Patch --> iBOT_Student["学生Patch头<br/>MLP"]
    Teacher_Patch --> iBOT_Teacher["教师Patch头<br/>MLP"]
    
    iBOT_Student --> iBOT_Loss["iBOT损失<br/>局部特征学习"]
    iBOT_Teacher --> iBOT_Loss
    
    %% KoLeo分支
    Student_Output --> KoLeo_Loss["KoLeo损失<br/>特征多样性"]
    
    DINO_Loss --> Total_Loss["总损失<br/>λ₁×DINO + λ₂×iBOT + λ₃×KoLeo"]
    iBOT_Loss --> Total_Loss
    KoLeo_Loss --> Total_Loss
    
    Total_Loss --> Backprop["反向传播<br/>(仅更新学生网络)"]
    Backprop --> EMA_Update["EMA更新<br/>教师←学生 (τ=0.996)"]
```

## 二、Vision Transformer with Register Tokens

```mermaid
graph TB
    Image["输入图像<br/>B×3×224×224"]
    
    Image --> PatchEmbed["Patch嵌入<br/>16×16 patches<br/>B×196×768"]
    
    PatchEmbed --> AddCLS["添加CLS Token<br/>B×197×768"]
    
    AddCLS --> AddRegister["添加Register Tokens<br/>B×(197+M)×768<br/>M=4 (推荐)"]
    
    AddRegister --> PosEmbed["位置编码<br/>可学习参数"]
    
    PosEmbed --> Tokens["完整Token序列<br/>CLS, REG₁, REG₂, REG₃, REG₄,<br/>PATCH₁, PATCH₂, ..., PATCH₁₉₆"]
    
    Tokens --> Block1["Transformer Block 1<br/>Self-Attention + FFN"]
    Block1 --> Block2["Transformer Block 2<br/>LayerScale + DropPath"]
    Block2 --> Block3["..."]
    Block3 --> BlockN["Transformer Block N<br/>(N = 12/24/40)"]
    
    BlockN --> Split{输出分离}
    
    Split --> CLS_Out["CLS Token输出<br/>B×768<br/>用于全局表示"]
    Split --> Register_Out["Register Token输出<br/>B×4×768<br/>训练后丢弃"]
    Split --> Patch_Out["Patch Token输出<br/>B×196×768<br/>用于密集预测"]
    
    CLS_Out --> Global_Head["全局投影头"]
    Patch_Out --> Patch_Head["Patch投影头"]
    Register_Out --> Discard["丢弃<br/>(仅训练时使用)"]
```

## 三、自监督学习流程详解

```mermaid
graph LR
    Original["原始图像<br/>3×H×W"]
    
    Original --> MultiCrop["多视角裁剪"]
    
    MultiCrop --> Global1["全局视角1<br/>3×224×224<br/>缩放: 0.32-1.0"]
    MultiCrop --> Global2["全局视角2<br/>3×224×224<br/>缩放: 0.32-1.0"]
    MultiCrop --> Local1["局部视角1<br/>3×96×96<br/>缩放: 0.05-0.32"]
    MultiCrop --> Local2["局部视角2<br/>3×96×96<br/>缩放: 0.05-0.32"]
    
    Global1 --> Teacher["教师网络<br/>ViT-g/14"]
    Global2 --> Teacher
    
    Global1 --> Student["学生网络<br/>ViT-g/14"]
    Global2 --> Student
    Local1 --> Student
    Local2 --> Student
    
    Teacher --> T_CLS["教师CLS<br/>全局特征"]
    Teacher --> T_Patch["教师Patch<br/>局部特征"]
    
    Student --> S_CLS["学生CLS<br/>全局特征"]
    Student --> S_Patch["学生Patch<br/>局部特征"]
    
    T_CLS --> DINO_T["DINO教师<br/>中心化+温度调节"]
    S_CLS --> DINO_S["DINO学生<br/>温度调节"]
    
    DINO_T --> Loss1["DINO损失<br/>KL散度"]
    DINO_S --> Loss1
    
    T_Patch --> iBOT_T["iBOT教师<br/>完整Patch"]
    S_Patch --> iBOT_S["iBOT学生<br/>遮挡Patch"]
    
    iBOT_T --> Loss2["iBOT损失<br/>遮挡重建"]
    iBOT_S --> Loss2
    
    S_CLS --> Loss3["KoLeo损失<br/>特征多样性"]
```

## 四、Register Tokens 注意力机制

```mermaid
graph TB
    subgraph Standard["标准ViT (无Register)"]
        Input1["CLS, P₁, P₂, ..., P₁₉₆"]
        Input1 --> Attn1["自注意力<br/>容易注意力坍塌"]
        Attn1 --> Problem["❌ 注意力集中<br/>❌ 特征单一"]
    end
    
    subgraph Enhanced["DINOv2 (有Register)"]
        Input2["CLS, R₁, R₂, R₃, R₄, P₁, P₂, ..., P₁₉₆"]
        Input2 --> Attn2["自注意力<br/>Register吸收多余信息"]
        Attn2 --> Success["✅ 注意力分散<br/>✅ 特征丰富"]
    end
    
    subgraph Distribution["注意力权重分布"]
        CLS["CLS Token"]
        Register["Register Tokens"]
        Patches["Patch Tokens"]
        
        CLS --> CLS_Attn["关注语义信息<br/>权重分散"]
        Register --> Reg_Attn["承接冗余信息<br/>防止坍塌"]
        Patches --> Patch_Attn["局部空间特征<br/>保持细节"]
    end
```

## 五、损失函数计算流程

```mermaid
graph TB
    subgraph DINO["DINO Loss (全局)"]
        S_Global["学生全局特征<br/>B×D"]
        T_Global["教师全局特征<br/>B×D"]
        
        T_Global --> Center_Update["中心化<br/>c = 0.9×c + 0.1×mean(T)"]
        Center_Update --> T_Centered["T - c"]
        
        T_Centered --> T_Softmax["Softmax(T/τₜ)<br/>τₜ=0.04~0.07"]
        S_Global --> S_LogSoftmax["LogSoftmax(S/τₛ)<br/>τₛ=0.1"]
        
        T_Softmax --> DINO_KL["KL散度<br/>-Σ T×log(S)"]
        S_LogSoftmax --> DINO_KL
    end
    
    subgraph iBOT["iBOT Loss (局部)"]
        S_Patches["学生Patch特征<br/>B×N×D (遮挡)"]
        T_Patches["教师Patch特征<br/>B×N×D (完整)"]
        Mask["遮挡掩码<br/>B×N"]
        
        S_Patches --> S_Patch_Proj["Patch投影<br/>MLP"]
        T_Patches --> T_Patch_Proj["Patch投影<br/>MLP"]
        
        S_Patch_Proj --> iBOT_KL["KL散度<br/>仅遮挡位置"]
        T_Patch_Proj --> iBOT_KL
        Mask --> iBOT_KL
    end
    
    subgraph KoLeo["KoLeo Loss (多样性)"]
        S_Features["学生特征<br/>B×D"]
        S_Features --> Distance["计算两两距离<br/>cdist(S, S)"]
        Distance --> KNN["k近邻距离<br/>k=2"]
        KNN --> Diversity["-log(knn_dist)<br/>鼓励特征分散"]
    end
    
    DINO_KL --> Combine["加权组合<br/>1.0×DINO + 1.0×iBOT + 0.1×KoLeo"]
    iBOT_KL --> Combine
    Diversity --> Combine
    
    Combine --> Total["总损失"]
```

## 六、训练数据处理管道

```mermaid
graph LR
    Internet["互联网图像<br/>Raw数据"]
    
    Internet --> Crawl["数据爬取<br/>~500M图像"]
    
    Crawl --> Filter1["质量筛选<br/>分辨率/清晰度"]
    
    Filter1 --> Dedup["智能去重<br/>感知哈希+特征相似度"]
    
    Dedup --> Filter2["内容过滤<br/>NSFW检测"]
    
    Filter2 --> Balance["域平衡<br/>确保多样性"]
    
    Balance --> Final["最终数据集<br/>142M高质量图像"]
    
    Final --> Batch["批次处理<br/>全局批次2048"]
    
    Batch --> Augment["数据增强<br/>RandomCrop+ColorJitter+Blur"]
    
    Augment --> Model["模型训练"]
    
    subgraph Auto["自动化流程"]
        Quality["质量评估<br/>美学分数"]
        Similarity["相似度计算<br/>CLIP特征"]
        Diversity["多样性度量<br/>聚类分析"]
    end
    
    Quality --> Filter1
    Similarity --> Dedup
    Diversity --> Balance
```

## 七、工程优化架构

```mermaid
graph TB
    subgraph Scale["大规模训练优化"]
        Model["ViT-g (1.1B参数)"]
        
        Model --> FSDP["FSDP分片<br/>参数+梯度+优化器"]
        FSDP --> Shard1["GPU 1<br/>参数分片 1/N"]
        FSDP --> Shard2["GPU 2<br/>参数分片 2/N"]
        FSDP --> ShardN["GPU N<br/>参数分片 N/N"]
        
        Shard1 --> Compute1["前向计算<br/>动态参数收集"]
        Shard2 --> Compute2["前向计算<br/>动态参数收集"]
        ShardN --> ComputeN["前向计算<br/>动态参数收集"]
        
        Compute1 --> AllReduce["AllReduce<br/>梯度同步"]
        Compute2 --> AllReduce
        ComputeN --> AllReduce
        
        AllReduce --> Update["参数更新<br/>分片优化器"]
    end
    
    subgraph Memory["内存优化"]
        Attention["注意力计算"]
        Attention --> xFormers["xFormers优化<br/>O(√N)内存复杂度"]
        
        Blocks["Transformer Blocks"]
        Blocks --> Chunking["Block Chunking<br/>梯度检查点"]
        
        Loss_Calc["损失计算"]
        Loss_Calc --> GradScale["梯度缩放<br/>FP16训练"]
    end
    
    subgraph Config["配置管理"]
        ConfigFile["YAML配置"]
        ConfigFile --> Student["学生网络配置"]
        ConfigFile --> Teacher["教师网络配置"]
        ConfigFile --> Training["训练超参数"]
    end
```

## 八、推理流程

```mermaid
graph LR
    Image["输入图像<br/>1×3×224×224"]
    
    Image --> Preprocess["预处理<br/>归一化+resize"]
    
    Preprocess --> ViT["ViT主干<br/>预训练权重"]
    
    ViT --> Tokens["Token输出<br/>CLS, REG, PATCH"]
    
    Tokens --> Extract{特征提取}
    
    Extract -->|分类任务| CLS["CLS特征<br/>1×768<br/>→ 线性分类器"]
    Extract -->|密集预测| Patches["Patch特征<br/>1×196×768<br/>→ 密集预测头"]
    Extract -->|丢弃| Registers["Register特征<br/>训练时使用"]
    
    CLS --> Classification["图像分类<br/>k-NN/线性探测"]
    Patches --> Dense["密集预测<br/>分割/深度估计"]
    
    Classification --> Result1["分类结果<br/>类别+置信度"]
    Dense --> Result2["像素级结果<br/>分割掩码/深度图"]
```

## 九、关键维度变化总结

```mermaid
graph TD
    Input["B×3×224×224<br/>输入图像"]
    
    Input -->|Patch嵌入| Patches["B×196×768<br/>Patch序列"]
    
    Patches -->|+CLS+Register| Tokens["B×201×768<br/>完整Token序列<br/>(1 CLS + 4 REG + 196 PATCH)"]
    
    Tokens -->|Transformer| Features["B×201×768<br/>编码特征"]
    
    Features -->|分离| Split{特征分离}
    
    Split -->|全局| CLS_Features["B×768<br/>CLS特征"]
    Split -->|寄存器| REG_Features["B×4×768<br/>Register特征"]
    Split -->|局部| PATCH_Features["B×196×768<br/>Patch特征"]
    
    CLS_Features -->|投影| CLS_Proj["B×65536<br/>DINO输出"]
    PATCH_Features -->|投影| PATCH_Proj["B×196×65536<br/>iBOT输出"]
    REG_Features -->|丢弃| Discard["训练后丢弃"]
    
    CLS_Proj --> Loss_Global["全局损失"]
    PATCH_Proj --> Loss_Patch["局部损失"]
    
    Loss_Global --> Total["总损失"]
    Loss_Patch --> Total
```

## 十、与DINOv1对比流程

```mermaid
graph LR
    subgraph V1["DINOv1 流程"]
        Input1["ImageNet-1k<br/>1.3M图像"]
        Input1 --> ViT1["ViT-B/16<br/>86M参数"]
        ViT1 --> DINO1["仅DINO损失<br/>全局特征"]
        DINO1 --> Result1["80.1% ImageNet<br/>研究原型"]
    end
    
    subgraph V2["DINOv2 流程"]
        Input2["自建数据集<br/>142M图像"]
        Input2 --> ViT2["ViT-g/14 + Register<br/>1.1B参数"]
        ViT2 --> Mixed["混合损失<br/>DINO+iBOT+KoLeo"]
        Mixed --> Result2["87.1% ImageNet<br/>工业就绪"]
    end
    
    subgraph Diff["关键差异"]
        Diff1["数据规模: 100×提升"]
        Diff2["模型规模: 11×提升"]  
        Diff3["架构创新: Register Tokens"]
        Diff4["训练策略: 混合损失函数"]
        Diff5["工程优化: FSDP+xFormers"]
    end
    
    Result1 -.->|升级| Result2
```

这些Mermaid流程图全面展示了DINOv2的核心架构、训练流程和关键创新点，包括Register Tokens机制、混合损失函数设计、大规模数据处理管道和工程优化方案。每个图表都详细标注了关键的维度变化和处理步骤，便于深入理解DINOv2的技术实现。