```
mini-llama/                           # 项目根目录
├── README.md                         # 项目总说明
├── requirements.txt                  # 依赖清单
├── configs/                          # 配置文件目录
│   ├── model_config.yaml             # 模型架构参数（hidden_size、num_heads等）
│   ├── train_config.yaml             # 训练参数（学习率、batch_size、轮次等）
│   └── data_config.yaml              # 数据参数（语料路径、序列长度等）
├── data/                             # 数据目录
│   ├── raw/                          # 原始语料（未预处理）
│   │   └── sample_corpus.txt         # 小规模测试语料（100-500MB）
│   ├── processed/                    # 预处理后语料（分词、编码后）
│   │   ├── train_ids.npy             # 训练集token id
│   │   └── val_ids.npy               # 验证集token id
│   └── tokenizer/                    # 自定义分词器相关文件
│       ├── tokenizer.json            # 训练好的BPE分词器模型
│       └── vocab.json                # 词汇表
├── src/                              # 核心代码目录
│   ├── __init__.py                   # 初始化
│   ├── tokenizer/                    # 分词器模块
│   │   ├── __init__.py
│   │   └── bpe_tokenizer.py          # 自定义BPE分词器实现
│   ├── model/                        # 模型核心组件
│   │   ├── __init__.py
│   │   ├── rope.py                   # RoPE位置编码实现
│   │   ├── rms_norm.py               # RMSNorm归一化实现
│   │   ├── swi_glu.py                # SwiGLU激活函数实现
│   │   ├── multi_head_attention.py   # 多头注意力+因果掩码实现
│   │   ├── decoder_layer.py          # 单层Decoder实现
│   │   └── decoder_model.py          # 完整Decoder-only模型组装
│   ├── data/                         # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_preprocess.py        # 语料清洗、预处理
│   │   └── data_loader.py            # 训练数据加载器（生成自回归样本）
│   ├── train/                        # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py                # 训练循环（前向/反向、日志记录）
│   │   └── optim_scheduler.py        # 优化器+学习率调度器
│   └── generate/                     # 文本生成模块（验证模型效果）
│       ├── __init__.py
│       └── text_generator.py         # 自回归文本生成实现
├── logs/                             # 训练日志目录（可追溯训练过程）
│   ├── train.log                     # 训练控制台日志（loss、PPL等）
│   └── tensorboard/                  # 可视化日志
├── checkpoints/                      # 模型权重保存目录
│   ├── model_epoch_n.pth             # 第n轮训练权重
│   └── best_model.pth                # 验证集PPL最优的模型权重
├── results/                          # 结果输出目录
│   ├── train_curves.png              # 训练loss/PPL曲线可视化
│   └── sample_generations.txt        # 模型生成的文本示例
└── scripts/
    ├──prepare_data.sh      # 数据预处理一键脚本（分词器训练+语料预处理）
    ├──train.sh             # 模型训练一键脚本
    ├──generate_text.sh     # 文本生成一键脚本
    └──utils.sh             # 通用工具函数（日志、参数校验，被其他脚本调用）
```