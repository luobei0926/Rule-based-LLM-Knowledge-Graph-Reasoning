import os
import torch
import pickle
from transformers import AutoTokenizer, AutoModel

# 设置类别实体目录路径
classified_entity_dir = "/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/BackTrack-master/data/chatdoctor5k/ClassifiedEntity"
output_dir = "/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/BackTrack-master/data/chatdoctor5k/EncodedEntity"

# 创建保存编码结果的目录
os.makedirs(output_dir, exist_ok=True)


# Mean Pooling - 考虑 attention mask 进行平均池化
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # 第一个元素包含所有 token 的嵌入
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# 加载模型
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained('/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/model/sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/model/sentence-transformers/all-mpnet-base-v2')
# 如果有 GPU，可以启用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("模型加载完成")

# 对每个类别的实体进行编码
for file_name in os.listdir(classified_entity_dir):
    # 只处理 .txt 文件
    if not file_name.endswith(".txt"):
        continue

    category = file_name.replace(".txt", "")  # 获取类别名称
    file_path = os.path.join(classified_entity_dir, file_name)

    # 读取实体
    with open(file_path, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]

    # 初始化嵌入列表
    embeddings = []

    print(f"正在处理类别：{category}，包含 {len(keywords)} 个实体...")

    # 对每个实体进行编码
    for keyword in keywords:
        # 将实体转换为模型输入
        encoded_input = tokenizer(keyword, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        # 获取模型输出并进行 mean pooling
        with torch.no_grad():
            model_output = model(**encoded_input)
            entity_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings.append(entity_embedding.cpu().numpy())  # 转为 NumPy 格式并添加到列表

    # 将嵌入结果转为 NumPy 数组
    embeddings = torch.tensor(embeddings).squeeze(1)

    # 创建关键词和嵌入的字典
    keyword_emb_dict = {
        "entities": keywords,
        "embeddings": embeddings.cpu().numpy(),  # 将嵌入转为 NumPy 数组
    }

    # 保存当前类别的关键词和嵌入到 pickle 文件
    output_file = os.path.join(output_dir, f"{category}_embeddings.pkl")
    print(f"Saving embeddings to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(keyword_emb_dict, f)

print("所有类别的实体编码已完成！")
