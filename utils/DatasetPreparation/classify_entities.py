import os

# 文件路径
train_file = "/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/BackTrack-master/data/chatdoctor5k/train.txt"
schema_file = "/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/BackTrack-master/data/chatdoctor5k/GraphKnowledge/schema.txt"
output_dir = "/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/BackTrack-master/data/chatdoctor5k/ClassifiedEntity"

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 读取 schema.txt，构建关系到分类的映射
relation_to_categories = {}
with open(schema_file, "r", encoding="utf-8") as schema_f:
    for line in schema_f:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        entity1_category, relation, entity2_category = line.split("-")
        relation_to_categories[relation] = (entity1_category, entity2_category)

# 初始化分类结果字典
classified_entities = {category: set() for category_pair in relation_to_categories.values() for category in
                       category_pair}

# 读取 train.txt 并根据关系分类实体
with open(train_file, "r", encoding="utf-8") as train_f:
    for line in train_f:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        entity1, relation, entity2 = line.split("\t")

        # 根据关系分类
        if relation in relation_to_categories:
            entity1_category, entity2_category = relation_to_categories[relation]
            classified_entities[entity1_category].add(entity1)
            classified_entities[entity2_category].add(entity2)

# 将分类后的实体写入到不同的文件
for category, entities in classified_entities.items():
    output_file = os.path.join(output_dir, f"{category}.txt")
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(sorted(entities)))  # 按字典顺序写入文件

print(f"实体分类完成，结果已保存到 {output_dir}")
