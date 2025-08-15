# DINO 数据流程图（Mermaid版）

> 注：如果某些图表无法正常显示，可以使用支持Mermaid的Markdown编辑器（如VSCode + Mermaid插件、Typora等）查看

## 一、整体架构流程图

```mermaid
graph TB
    Input["输入图像<br/>[B, 3, H, W]"]
    
    Input --> Backbone["Backbone (ResNet/Swin)<br/>多尺度特征提取"]
    
    Backbone --> MS_Features["多尺度特征<br/>[B, C3, H/8, W/8]<br/>[B, C4, H/16, W/16]<br/>[B, C5, H/32, W/32]<br/>[B, C6, H/64, W/64]"]
    
    MS_Features --> Projection["特征投影<br/>统一到256维"]
    
    Projection --> Unified_Features["统一维度特征<br/>[B, 256, H_i, W_i] × 4"]
    
    Unified_Features --> Encoder["Transformer编码器<br/>6层"]
    
    Encoder --> Memory["编码器输出<br/>[B, ΣHW, 256]<br/>约20000个tokens"]
    
    Memory --> TopK["Top-K选择<br/>(Mixed Query Selection)"]
    
    TopK --> Selected["选中的查询<br/>[B, 900, 256]"]
    
    Selected --> Decoder_Input["解码器输入准备"]
    
    %% CDN分支（训练时）
    Training_GT["GT标签和框<br/>(训练时)"] --> CDN["CDN准备<br/>生成噪声查询"]
    CDN --> DN_Queries["去噪查询<br/>[B, 2000, 256]<br/>正样本×1000<br/>负样本×1000"]
    DN_Queries --> Decoder_Input
    CDN --> Mask["注意力掩码<br/>[2900, 2900]"]
    Mask --> Decoder
    
    Decoder_Input --> Combined["合并查询<br/>[B, 2900, 256]<br/>去噪2000+匹配900"]
    
    Combined --> Decoder["Transformer解码器<br/>6层迭代优化"]
    
    Decoder --> Dec_Output["解码器输出<br/>[6, B, 2900, 256]"]
    
    Dec_Output --> Split{分离输出}
    
    Split -->|训练时| DN_Out["CDN输出<br/>[B, 2000, 91+4]"]
    Split --> Match_Out["匹配输出<br/>[B, 900, 91+4]"]
    
    Match_Out --> Final["最终预测<br/>类别: [B, 900, 91]<br/>边界框: [B, 900, 4]"]
    
    DN_Out --> Loss_CDN["CDN损失"]
    Final --> Loss_Match["匹配损失"]
    
    Loss_CDN --> Total_Loss["总损失"]
    Loss_Match --> Total_Loss
```

## 二、CDN详细流程

```mermaid
graph LR
    GT["GT框和标签<br/>[N_gt, 4+1]"]
    
    GT --> Repeat["重复2×100次<br/>生成2000个样本"]
    
    Repeat --> Split_PN{正负样本分组}
    
    Split_PN -->|前1000个| Positive["正样本组<br/>索引: [0,1,2,...,999]"]
    Split_PN -->|后1000个| Negative["负样本组<br/>索引: [1000,1001,...,1999]"]
    
    Positive --> Small_Noise["添加小噪声<br/>λ₁ = 0.4<br/>噪声∈[-0.4s, 0.4s]"]
    Negative --> Large_Noise["添加大噪声<br/>λ₂ = 1.0<br/>噪声∈[0.4s, 1.0s]"]
    
    Small_Noise --> Pos_Query["正样本查询<br/>目标: 重构GT框"]
    Large_Noise --> Neg_Query["负样本查询<br/>目标: 预测背景"]
    
    Pos_Query --> DN_Queries["去噪查询集合<br/>[B, 2000, 256]"]
    Neg_Query --> DN_Queries
    
    DN_Queries --> Attention_Mask["生成注意力掩码<br/>组内可见<br/>组间隔离"]
```

## 三、编码器处理流程

```mermaid
graph TB
    Features["多尺度特征<br/>[B, 256, H_i, W_i] × 4"]
    
    Features --> Flatten["展平操作<br/>flatten(2).transpose(1,2)"]
    
    Flatten --> Tokens["Token序列<br/>L1: H/8×W/8 ≈ 15000<br/>L2: H/16×W/16 ≈ 3750<br/>L3: H/32×W/32 ≈ 950<br/>L4: H/64×W/64 ≈ 247"]
    
    Tokens --> Concat["拼接所有层<br/>[B, ≈20000, 256]"]
    
    Concat --> Enc_Layer1["编码器层1<br/>Self-Attention"]
    Enc_Layer1 --> Enc_Layer2["编码器层2<br/>Self-Attention"]
    Enc_Layer2 --> Enc_Layer3["..."]
    Enc_Layer3 --> Enc_Layer6["编码器层6<br/>Self-Attention"]
    
    Enc_Layer6 --> Enhanced["增强特征<br/>[B, ≈20000, 256]"]
    
    Enhanced --> Proposals["生成提议框<br/>gen_encoder_output_proposals"]
    
    Proposals --> Class_Head["分类预测<br/>[B, ≈20000, 91]"]
    Proposals --> Box_Head["边界框预测<br/>[B, ≈20000, 4]"]
    
    Class_Head --> Scores["置信度分数<br/>max(-1)"]
    Scores --> TopK_Select["Top-K选择<br/>K=900"]
    
    TopK_Select --> Selected_Indices["选中索引<br/>[B, 900]"]
    
    Selected_Indices --> Gather["收集特征"]
    Enhanced --> Gather
    Box_Head --> Gather
    
    Gather --> Output["编码器输出<br/>内容: [B, 900, 256]<br/>位置: [B, 900, 4]"]
```

## 四、解码器层内处理（Look Forward Twice）

```mermaid
graph TB
    Input["层输入<br/>Query: [2900, B, 256]<br/>Ref: [2900, B, 4]"]
    
    Input --> SA["自注意力<br/>带CDN掩码"]
    
    SA --> CA["交叉注意力<br/>可变形注意力机制"]
    
    CA --> FFN["前馈网络<br/>2048维"]
    
    FFN --> Features["特征输出<br/>[2900, B, 256]"]
    
    Features --> BBox_Embed["边界框嵌入层<br/>MLP"]
    
    BBox_Embed --> Delta["预测偏移<br/>Δb"]
    
    Input --> Ref_Unsig["inverse_sigmoid<br/>参考点"]
    
    Delta --> Add["相加<br/>Δb + ref"]
    Ref_Unsig --> Add
    
    Add --> Sigmoid["sigmoid"]
    
    Sigmoid --> New_Ref["新参考点<br/>[2900, B, 4]"]
    
    New_Ref --> Branch{Look Forward Twice}
    
    Branch -->|用于下一层| Detach["detach()<br/>断开梯度"]
    Branch -->|用于损失| Keep_Grad["保留梯度"]
    
    Detach --> Next_Layer["下一层输入"]
    Keep_Grad --> Loss_Calc["损失计算"]
    
    Features --> Layer_Out["层输出"]
```

## 五、损失计算流程

```mermaid
graph TB
    Predictions["模型预测"]
    Targets["真实标签"]
    
    Predictions --> Split_Pred{分离预测}
    Split_Pred --> Match_Pred["匹配预测<br/>[B, 900, 91+4]"]
    Split_Pred --> DN_Pred["去噪预测<br/>[B, 2000, 91+4]"]
    
    Match_Pred --> Hungarian["匈牙利匹配"]
    Targets --> Hungarian
    
    Hungarian --> Indices["匹配索引<br/>(src_idx, tgt_idx)"]
    
    Indices --> Match_Loss["匹配损失计算"]
    Match_Pred --> Match_Loss
    Targets --> Match_Loss
    
    Match_Loss --> L_cls["分类损失<br/>Focal Loss"]
    Match_Loss --> L_box["边界框损失<br/>L1 + GIOU"]
    
    DN_Pred --> Split_DN{正负样本分离}
    Split_DN --> Pos_Pred["正样本预测<br/>[B, 1000, 91+4]"]
    Split_DN --> Neg_Pred["负样本预测<br/>[B, 1000, 91+4]"]
    
    Pos_Pred --> Pos_Loss["正样本损失<br/>重构GT"]
    Targets --> Pos_Loss
    
    Neg_Pred --> Neg_Loss["负样本损失<br/>预测背景"]
    
    L_cls --> Weight1["×1.0"]
    L_box --> Weight2["×5.0 (L1)<br/>×2.0 (GIOU)"]
    Pos_Loss --> Weight3["×1.0"]
    Neg_Loss --> Weight4["×1.0"]
    
    Weight1 --> Total["总损失"]
    Weight2 --> Total
    Weight3 --> Total
    Weight4 --> Total
    
    Total --> Backprop["反向传播"]
```

## 六、推理流程（简化版）

```mermaid
graph LR
    Image["输入图像<br/>[1, 3, H, W]"]
    
    Image --> Feature["特征提取<br/>Backbone+Encoder"]
    
    Feature --> TopK["Top-900选择"]
    
    TopK --> Decoder["解码器<br/>无CDN查询"]
    
    Decoder --> Pred["预测<br/>[1, 900, 91+4]"]
    
    Pred --> NMS["NMS后处理"]
    
    NMS --> Result["检测结果<br/>[(x,y,w,h,cls,score), ...]"]
```

## 七、关键维度变化总结

```mermaid
graph TD
    Start["[B, 3, H, W]<br/>原始图像"]
    
    Start -->|Backbone| F1["[B, 256, H_i, W_i]×4<br/>多尺度特征"]
    
    F1 -->|Flatten| F2["[B, ≈20000, 256]<br/>展平tokens"]
    
    F2 -->|Encoder| F3["[B, ≈20000, 256]<br/>增强特征"]
    
    F3 -->|Top-K| F4["[B, 900, 256]<br/>选中查询"]
    
    F4 -->|+CDN| F5["[B, 2900, 256]<br/>训练时合并"]
    
    F5 -->|Decoder| F6["[6, B, 2900, 256]<br/>6层输出"]
    
    F6 -->|Split| F7["匹配: [B, 900, 91+4]<br/>CDN: [B, 2000, 91+4]"]
    
    F7 -->|Loss| F8["标量损失值"]
```

这些Mermaid图表清晰地展示了DINO模型的完整数据流程，包括关键的维度变化、处理步骤和模块间的连接关系。