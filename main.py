from transformers import pipeline
import wikipediaapi
import torch
import numpy as np

# 加载NER（命名实体识别）模型
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# 初始化 Wikipedia API
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='Test_demo(Vrije Universiteit in Amsterdam)/1.0 (s.wang@student.vu.nl)'
)

def get_wikipedia_link(entity):
    # 尝试查找实体的 Wikipedia 页面
    page = wiki.page(entity)
    if page.exists():
        return page.fullurl
    return None


def extract_entities(text):
    # 使用NER模型进行预测
    entities = ner_pipeline(text)
    # 格式化输出
    extracted_entities = [{"entity": ent["entity_group"], "word": ent["word"], "score": ent["score"]} for ent in
                          entities]
    return extracted_entities


def extract_entities_with_wiki(text):
    # 使用 NER 模型识别实体
    entities = ner_pipeline(text)
    # 为每个实体查找 Wikipedia 页面链接
    result = {}
    for ent in entities:
        entity_name = ent["word"]
        wiki_url = get_wikipedia_link(entity_name)
        if wiki_url:
            result[entity_name] = wiki_url
    return result


text = (
    "Is Managua the capital of Nicaragua? ")
# entities = extract_entities(text)
# print(entities)
entities_with_links = extract_entities_with_wiki(text)
for entity, url in entities_with_links.items():
    print(f"{entity} ⇒ {url}")
