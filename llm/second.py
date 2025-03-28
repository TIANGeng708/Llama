import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from evaluate import load
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# === Step 1: 加载微调模型和分词器 ===
mode_path = "fine_tuned_model"  # 替换为你的微调模型路径
tokenizer = AutoTokenizer.from_pretrained(mode_path)
model = AutoModelForQuestionAnswering.from_pretrained(mode_path)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Step 2: 加载本地 Parquet 数据集 ===
train_data_path = "./data/squad/train-00000-of-00001.parquet"
valid_data_path = "./data/squad/validation-00000-of-00001.parquet"

# 加载验证集
dataset = load_dataset('parquet', data_files={'validation': valid_data_path}, split='validation')

# === Step 3: 定义预测函数 ===
def predict(context, question):
    # 编码输入
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取起始位置和结束位置的索引
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits).item()
    end_index = torch.argmax(end_logits).item()
    
    # 解码答案
    answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer

# === Step 4: 遍历数据集并进行预测 ===
metric = load("squad")  # 加载评估器

predictions = []
references = []

for sample in dataset:
    context = sample["context"]
    question = sample["question"]
    answers = sample["answers"]
    
    # 使用模型预测答案
    predicted_answer = predict(context, question)
    
    # 存储预测结果和参考答案
    predictions.append({"id": sample["id"], "prediction_text": predicted_answer})
    references.append({"id": sample["id"], "answers": {"text": answers["text"], "answer_start": answers["answer_start"]}})
    
    # 打印部分结果（可选）
    print(f"Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")
    print(f"True Answer: {answers['text'][0]}")
    print("-" * 50)

# === Step 5: 计算 F1 和 EM 分数 ===
results = metric.compute(predictions=predictions, references=references)
print(f"F1 Score: {results['f1']}")
print(f"Exact Match (EM) Score: {results['exact_match']}")