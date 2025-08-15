# DINO自监督学习算法流程图集合

## 1. 整体架构流程图

```mermaid
flowchart TD
    Input["输入图像<br/>[H×W×3]"]
    
    %% 数据增强分组
    subgraph "多尺度数据增强"
        Aug1["全局增强1<br/>(224×224)"]
        Aug2["全局增强2<br/>(224×224)"]
        Aug3["局部增强1<br/>(96×96)"]
        Aug4["局部增强2<br/>(96×96)"]
        AugN["局部增强N<br/>(96×96)"]
    end
    
    Input --> Aug1
    Input --> Aug2
    Input --> Aug3
    Input --> Aug4
    Input --> AugN
    
    %% 网络前向传播
    subgraph "网络架构"
        Student["学生网络<br/>ViT + DINO Head"]
        Teacher["教师网络<br/>ViT + DINO Head"]
    end
    
    Aug1 --> Student
    Aug2 --> Student
    Aug3 --> Student
    Aug4 --> Student
    AugN --> Student
    
    Aug1 --> Teacher
    Aug2 --> Teacher
    
    %% 损失计算和参数更新
    Student --> SOutput["学生输出<br/>[B×N_crops, K]"]
    Teacher --> TOutput["教师输出<br/>[B×2, K]"]
    
    SOutput --> Loss["DINO损失计算<br/>交叉熵 + 中心化"]
    TOutput --> Loss
    
    Loss --> BackProp["反向传播<br/>→ 学生网络"]
    TOutput --> EMA["EMA更新<br/>→ 教师网络"]
    
    %% 样式
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef augStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef networkStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef outputStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef updateStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class Input inputStyle
    class Aug1,Aug2,Aug3,Aug4,AugN augStyle
    class Student,Teacher networkStyle
    class SOutput,TOutput,Loss outputStyle
    class BackProp,EMA updateStyle
```

## 2. 数据增强和多尺度裁剪流程图

```mermaid
graph LR
    Image["原始图像<br/>[H×W×3]"]
    
    %% 全局裁剪分支
    Image --> Global["全局裁剪<br/>(0.4-1.0 scale)"]
    Global --> GResize1["调整大小<br/>224×224"]
    Global --> GResize2["调整大小<br/>224×224"]
    GResize1 --> GAug1["随机增强<br/>(翻转、色彩变换)"]
    GResize2 --> GAug2["随机增强<br/>(翻转、色彩变换)"]
    
    %% 局部裁剪分支
    Image --> Local["局部裁剪<br/>(0.05-0.4 scale)"]
    Local --> LResize1["调整大小<br/>96×96"]
    Local --> LResize2["调整大小<br/>96×96"]
    Local --> LResizeN["...更多局部裁剪<br/>96×96"]
    LResize1 --> LAug1["随机增强<br/>(翻转、色彩变换)"]
    LResize2 --> LAug2["随机增强<br/>(翻转、色彩变换)"]
    LResizeN --> LAugN["随机增强<br/>(翻转、色彩变换)"]
    
    %% 输出
    GAug1 --> TInput["教师输入<br/>[2, 3, 224, 224]"]
    GAug2 --> TInput
    
    GAug1 --> SInput["学生输入<br/>[N_crops, 3, 224/96, 224/96]"]
    GAug2 --> SInput
    LAug1 --> SInput
    LAug2 --> SInput
    LAugN --> SInput
    
    %% 样式
    classDef originalStyle fill:#ffebee
    classDef globalStyle fill:#e8f5e8
    classDef localStyle fill:#fff3e0
    classDef outputStyle fill:#e1f5fe
    
    class Image originalStyle
    class Global,GResize1,GResize2,GAug1,GAug2 globalStyle
    class Local,LResize1,LResize2,LResizeN,LAug1,LAug2,LAugN localStyle
    class TInput,SInput outputStyle
```

## 3. Vision Transformer前向传播流程图

```mermaid
flowchart TD
    %% 输入阶段
    Input["输入图像<br/>[B, 3, H, W]"]
    
    %% Patch Embedding阶段
    Input --> Patch["图像分块<br/>[B, N_patches, patch_size²×3]"]
    Patch --> Embed["线性嵌入<br/>[B, N_patches, embed_dim]"]
    
    %% 添加CLS Token和位置编码
    CLS["CLS Token<br/>[1, 1, embed_dim]"] 
    Embed --> Concat["拼接CLS Token<br/>[B, N_patches+1, embed_dim]"]
    CLS --> Concat
    Concat --> PosAdd["+ 位置编码<br/>[B, N_patches+1, embed_dim]"]
    
    %% Transformer Blocks主流程
    PosAdd --> Block1["Transformer Block 1"]
    Block1 --> Block2["Transformer Block 2"]
    Block2 --> BlockDots["..."]
    BlockDots --> BlockN["Transformer Block N"]
    
    %% 输出处理
    BlockN --> CLSOut["提取CLS Token<br/>[B, embed_dim]"]
    CLSOut --> DinoHead["DINO Head<br/>MLP + 归一化"]
    DinoHead --> FinalOut["最终输出<br/>[B, out_dim]"]
    
    %% 样式定义
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef embedStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef transformerStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef outputStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class Input inputStyle
    class Patch,Embed,CLS,Concat,PosAdd embedStyle
    class Block1,Block2,BlockDots,BlockN transformerStyle
    class CLSOut,DinoHead,FinalOut outputStyle
```

### 单个Transformer Block详细结构

```mermaid
flowchart TD
    BlockInput["输入 x<br/>[B, N+1, embed_dim]"]
    
    %% 第一个子层：多头自注意力
    BlockInput --> Norm1["LayerNorm"]
    Norm1 --> Attention["多头自注意力<br/>Q=xW_q, K=xW_k, V=xW_v"]
    
    %% 残差连接1
    BlockInput --> Add1["+"]
    Attention --> Add1
    Add1 --> Output1["输出1<br/>[B, N+1, embed_dim]"]
    
    %% 第二个子层：前馈网络
    Output1 --> Norm2["LayerNorm"]
    Norm2 --> MLP["MLP层<br/>Linear → GELU → Linear"]
    
    %% 残差连接2
    Output1 --> Add2["+"]
    MLP --> Add2
    Add2 --> BlockOutput["Block输出<br/>[B, N+1, embed_dim]"]
    
    %% 样式
    classDef normStyle fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef attnStyle fill:#fff3e0,stroke:#ff6f00,stroke-width:2px
    classDef mlpStyle fill:#f1f8e9,stroke:#388e3c,stroke-width:2px
    classDef addStyle fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    classDef ioStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    
    class Norm1,Norm2 normStyle
    class Attention attnStyle
    class MLP mlpStyle
    class Add1,Add2 addStyle
    class BlockInput,Output1,BlockOutput ioStyle
```

## 4. DINO损失计算流程图

```mermaid
flowchart TD
    %% 输入数据
    subgraph "网络输出"
        SOut["学生输出<br/>[B×N_crops, out_dim]"]
        TOut["教师输出<br/>[B×2, out_dim]"]
    end
    
    %% 学生输出处理分支
    subgraph "学生输出处理"
        STemp["温度缩放<br/>student_out / τ_s<br/>(τ_s = 0.1)"]
        SChunk["分块处理<br/>chunk(N_crops)"]
        SLogSoftmax["LogSoftmax<br/>准备计算损失"]
    end
    
    %% 教师输出处理分支  
    subgraph "教师输出处理"
        Center["中心化<br/>teacher_out - center"]
        TTemp["温度缩放<br/>(centered) / τ_t"]
        TSoftmax["Softmax归一化<br/>P_t = softmax(...)"]
        TChunk["分块处理<br/>chunk(2)"]
        TDetach["梯度截断<br/>detach()"]
    end
    
    %% 连接关系
    SOut --> STemp
    STemp --> SChunk
    SChunk --> SLogSoftmax
    
    TOut --> Center
    Center --> TTemp
    TTemp --> TSoftmax
    TSoftmax --> TChunk
    TChunk --> TDetach
    
    %% 损失计算核心
    subgraph "交叉熵损失计算"
        LossLoop["遍历所有view对<br/>(跳过相同view)"]
        CrossEntropy["交叉熵<br/>-P_t × log_softmax(P_s)"]
        SumLoss["累加所有损失项"]
        AvgLoss["平均损失<br/>total_loss / n_terms"]
    end
    
    SLogSoftmax --> LossLoop
    TDetach --> LossLoop
    LossLoop --> CrossEntropy
    CrossEntropy --> SumLoss
    SumLoss --> AvgLoss
    
    %% 中心更新
    subgraph "中心动态更新"
        UpdateCenter["center ← m×center + (1-m)×mean(T_out)<br/>m = 0.9 (momentum)"]
    end
    
    TOut --> UpdateCenter
    
    %% 最终输出
    AvgLoss --> FinalLoss["最终DINO损失"]
    UpdateCenter --> NewCenter["更新后的中心向量"]
    
    %% 样式定义
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef studentStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef teacherStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef lossStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef centerStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef outputStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class SOut,TOut inputStyle
    class STemp,SChunk,SLogSoftmax studentStyle
    class Center,TTemp,TSoftmax,TChunk,TDetach teacherStyle
    class LossLoop,CrossEntropy,SumLoss,AvgLoss lossStyle
    class UpdateCenter,NewCenter centerStyle
    class FinalLoss outputStyle
```

## 5. 训练流程图

```mermaid
graph TD
    Start["开始训练"]
    
    %% 初始化
    Start --> Init["初始化网络<br/>Student & Teacher"]
    Init --> LoadData["加载数据<br/>ImageNet"]
    
    %% 主训练循环
    LoadData --> EpochLoop{"Epoch循环<br/>(1 to epochs)"}
    EpochLoop --> BatchLoop{"Batch循环"}
    
    %% 单个batch的处理
    BatchLoop --> Schedule["更新调度器<br/>LR, WD, Momentum"]
    Schedule --> Forward["前向传播"]
    
    %% 前向传播详细
    Forward --> SForward["学生网络前向<br/>所有crops"]
    Forward --> TForward["教师网络前向<br/>仅全局crops"]
    
    %% 损失计算和反向传播
    SForward --> LossComp["计算DINO损失"]
    TForward --> LossComp
    LossComp --> CheckLoss{"损失是否有限?"}
    
    CheckLoss -->|否| StopTrain["停止训练<br/>打印错误"]
    CheckLoss -->|是| Backward["反向传播"]
    
    %% 参数更新
    Backward --> ClipGrad["梯度裁剪<br/>(可选)"]
    ClipGrad --> OptStep["优化器步骤<br/>更新学生网络"]
    OptStep --> EMAUpdate["EMA更新<br/>更新教师网络"]
    
    %% 循环控制
    EMAUpdate --> BatchEnd{"批次结束?"}
    BatchEnd -->|否| BatchLoop
    BatchEnd -->|是| EpochEnd{"轮次结束?"}
    
    %% 验证和保存
    EpochEnd -->|否| EpochLoop
    EpochEnd -->|是| Validate["验证模型<br/>(可选)"]
    Validate --> SaveModel["保存检查点"]
    SaveModel --> TrainEnd{"训练完成?"}
    
    TrainEnd -->|否| EpochLoop
    TrainEnd -->|是| End["训练结束"]
    
    %% EMA更新详细过程
    subgraph "EMA更新详细"
        direction LR
        StudentParam["学生参数 θ_s"]
        TeacherParam["教师参数 θ_t"]
        Momentum["动量 m"]
        
        StudentParam --> EMAFormula["θ_t ← m×θ_t + (1-m)×θ_s"]
        TeacherParam --> EMAFormula
        Momentum --> EMAFormula
        EMAFormula --> NewTeacher["新教师参数"]
    end
    
    %% 样式
    classDef startEndStyle fill:#e8f5e8
    classDef loopStyle fill:#fff3e0
    classDef processStyle fill:#e1f5fe
    classDef decisionStyle fill:#ffebee
    classDef errorStyle fill:#ff5252,color:#fff
    
    class Start,End startEndStyle
    class EpochLoop,BatchLoop,BatchEnd,EpochEnd,TrainEnd loopStyle
    class Init,LoadData,Schedule,Forward,SForward,TForward,LossComp,Backward,ClipGrad,OptStep,EMAUpdate,Validate,SaveModel processStyle
    class CheckLoss decisionStyle
    class StopTrain errorStyle
```

## 6. 推理和评估流程图

```mermaid
graph TB
    TestImg["测试图像<br/>[H×W×3]"]
    
    %% 预处理
    TestImg --> Preprocess["预处理<br/>Resize + Normalize"]
    Preprocess --> Batch["批处理<br/>[B, 3, 224, 224]"]
    
    %% 特征提取
    Batch --> Backbone["ViT Backbone<br/>特征提取"]
    Backbone --> CLSToken["CLS Token<br/>[B, embed_dim]"]
    
    %% 不同评估方式
    CLSToken --> KNN["k-NN分类"]
    CLSToken --> Linear["线性分类"]
    CLSToken --> Attention["注意力可视化"]
    
    %% k-NN分类详细
    subgraph "k-NN分类过程"
        direction TB
        KNNFeat["特征向量<br/>[embed_dim]"]
        KNNFeat --> Normalize1["L2归一化"]
        Normalize1 --> Distance["计算距离<br/>与训练集特征"]
        Distance --> TopK["选择Top-k<br/>最近邻"]
        TopK --> Vote["投票决策<br/>多数投票"]
        Vote --> KNNResult["k-NN结果"]
    end
    
    %% 线性分类详细
    subgraph "线性分类过程"
        direction TB
        LinearFeat["特征向量<br/>[embed_dim]"]
        LinearFeat --> LinearLayer["线性分类器<br/>W×x + b"]
        LinearLayer --> Softmax["Softmax<br/>概率分布"]
        Softmax --> LinearResult["分类结果"]
    end
    
    %% 注意力可视化详细
    subgraph "注意力可视化"
        direction TB
        AttnFeat["注意力权重<br/>[num_heads, N+1, N+1]"]
        AttnFeat --> CLSAttn["提取CLS注意力<br/>[num_heads, N]"]
        CLSAttn --> AvgHead["头平均<br/>[N]"]
        AvgHead --> Reshape["重塑为图像<br/>[H/16, W/16]"]
        Reshape --> Interpolate["插值上采样<br/>[H, W]"]
        Interpolate --> Heatmap["热力图可视化"]
    end
    
    %% 输出
    KNN --> KNNResult
    Linear --> LinearResult
    Attention --> Heatmap
    
    KNNResult --> Metrics["性能指标<br/>Accuracy, Top-5"]
    LinearResult --> Metrics
    Heatmap --> Visual["可视化结果<br/>注意力图"]
    
    %% 样式
    classDef inputStyle fill:#e1f5fe
    classDef processStyle fill:#fff3e0
    classDef methodStyle fill:#f3e5f5
    classDef outputStyle fill:#e8f5e8
    
    class TestImg,Batch inputStyle
    class Preprocess,Backbone,CLSToken processStyle
    class KNN,Linear,Attention,KNNFeat,Normalize1,Distance,TopK,Vote,LinearFeat,LinearLayer,Softmax,AttnFeat,CLSAttn,AvgHead,Reshape,Interpolate methodStyle
    class KNNResult,LinearResult,Heatmap,Metrics,Visual outputStyle
```

## 7. 数据维度变化图

```mermaid
flowchart LR
    %% 输入阶段
    subgraph Input["输入阶段"]
        direction TB
        OrigImg["原始图像<br/>[H×W×3]<br/>例: 256×256×3"]
        MultiCrop["多尺度裁剪<br/>[N_crops, 3, H', W']<br/>例: 10×3×224×224"]
        OrigImg --> MultiCrop
    end
    
    %% 嵌入阶段  
    subgraph Embed["嵌入阶段"]
        direction TB
        Patches["图像分块<br/>[N_crops, N_patches, P²×3]<br/>例: 10×196×768"]
        LinearEmbed["线性嵌入<br/>[N_crops, N_patches, D]<br/>例: 10×196×384"]
        AddCLS["+ CLS Token<br/>[N_crops, N_patches+1, D]<br/>例: 10×197×384"]
        PosEncode["+ 位置编码<br/>[N_crops, N_patches+1, D]<br/>例: 10×197×384"]
        
        Patches --> LinearEmbed
        LinearEmbed --> AddCLS
        AddCLS --> PosEncode
    end
    
    %% Transformer阶段
    subgraph Trans["Transformer阶段"]
        direction TB
        Block1["Block 1<br/>[N_crops, N+1, D]<br/>例: 10×197×384"]
        BlockDots["...<br/>维度保持不变"]
        BlockN["Block N<br/>[N_crops, N+1, D]<br/>例: 10×197×384"]
        
        Block1 --> BlockDots
        BlockDots --> BlockN
    end
    
    %% 输出阶段
    subgraph Output["输出阶段"]
        direction TB
        ExtractCLS["提取CLS Token<br/>[N_crops, D]<br/>例: 10×384"]
        DinoHead["DINO Head<br/>[N_crops, bottleneck]<br/>例: 10×256"]
        FinalProj["最终投影<br/>[N_crops, K]<br/>例: 10×65536"]
        
        ExtractCLS --> DinoHead
        DinoHead --> FinalProj
    end
    
    %% 流程连接
    Input --> Embed
    MultiCrop -.->|Patch分割 P=16| Patches
    Embed --> Trans
    PosEncode --> Block1
    Trans --> Output
    BlockN --> ExtractCLS
    
    %% 关键参数说明
    subgraph Params["关键参数"]
        direction TB
        P1["P = patch_size = 16<br/>分块大小"]
        P2["D = embed_dim = 384/768<br/>嵌入维度"] 
        P3["N_patches = 224/16² = 196<br/>分块数量"]
        P4["K = out_dim = 65536<br/>输出维度"]
        P5["N_crops = 2全局 + 8局部<br/>数据增强数量"]
    end
    
    %% 样式定义
    classDef inputStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef embedStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px  
    classDef transStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef outputStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef paramStyle fill:#fafafa,stroke:#616161,stroke-width:2px,stroke-dasharray: 3 3
    
    class Input,OrigImg,MultiCrop inputStyle
    class Embed,Patches,LinearEmbed,AddCLS,PosEncode embedStyle
    class Trans,Block1,BlockDots,BlockN transStyle
    class Output,ExtractCLS,DinoHead,FinalProj outputStyle
    class Params,P1,P2,P3,P4,P5 paramStyle
```

## 总结

这些流程图全面展示了DINO算法的各个关键环节：

1. **整体架构图**: 展示了教师-学生网络的完整训练框架
2. **数据增强图**: 详细说明了多尺度裁剪的实现方式
3. **ViT前向传播图**: 展示了Vision Transformer的详细计算过程
4. **损失计算图**: 说明了DINO损失函数的计算逻辑
5. **训练流程图**: 展示了完整的训练循环和参数更新过程
6. **推理评估图**: 说明了模型的不同评估方式
7. **维度变化图**: 追踪了数据在网络中的维度变化

这些图表有助于深入理解DINO算法的技术细节和实现原理。