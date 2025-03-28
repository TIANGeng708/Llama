# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from datasets import load_dataset
# from peft import get_peft_model, LoraConfig

# # ==================== 配置部分 ====================
# # 1. 选择显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # 如果你只有两张卡
# # device_ids = [0, 1]

# # 2. 本地模型路径（提前下载好）
# model_path = "./model/2.7B"

# # 3. 本地数据集路径（提前转换成parquet）
# train_data_path = "./data/squad/train-00000-of-00001.parquet"
# valid_data_path = "./data/squad/validation-00000-of-00001.parquet"

# # ==================== 加载模型和Tokenizer ====================
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)

# # GPT系列有时候没有pad_token，补上
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # 多GPU
# # model = torch.nn.DataParallel(model).cuda()
# lora_config = LoraConfig(
#     r=4,  # Rank of the low-rank decomposition,可以根据需要调节
#     lora_alpha=16,
#     lora_dropout=0.1,
#     bias="none",  # 是否使用偏置项
# )

# # 在模型中应用LoRA
# model = get_peft_model(model, lora_config)

# # 如果使用多GPU
# # device_ids = [0,1]  # 选择GPU
# # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
# # ==================== 加载本地数据集 ====================
# dataset = load_dataset("parquet", data_files={
#     "train": train_data_path,
#     "validation": valid_data_path
# })

# # ==================== 预处理 ====================
# def preprocess_function(examples):
#     inputs = tokenizer(examples["context"], examples["question"],
#                        truncation=True, padding="max_length", max_length=512)
#     # GPT这种Causal LM是自回归模型，labels直接是input_ids的shifted copy
#     inputs["labels"] = inputs["input_ids"].copy()
#     return inputs

# # 把所有列都删掉，换成tokenizer输出的列（input_ids, attention_mask, labels）
# tokenized_datasets = dataset.map(preprocess_function, batched=True,
#                                  remove_columns=dataset["train"].column_names)

# # ==================== 训练参数 ====================
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=500,
#     fp16=True,
#     save_steps=1000,
#     remove_unused_columns=False  # 【关键】绝对不能让Trainer自动删列
# )

# # 【关键】DataCollator要用语言模型专用的
# from transformers import DataCollatorForLanguageModeling
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False  # 自回归任务，不做mask
# )

# # Trainer初始化要带上collator
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     tokenizer=tokenizer,
#     data_collator=data_collator  # 【关键】语言模型的collator
# )


# # ==================== 开始训练 ====================
# trainer.train()

# # ==================== 保存模型 ====================
# # model.module.save_pretrained("./fine_tuned_model")  # DataParallel时要加.module
# model.save_pretrained("./fine_tuned_model") 
# tokenizer.save_pretrained("./fine_tuned_model")
