from transformers import pipeline, BertTokenizer, BertModel
import wikipedia
import torch
from scipy.spatial.distance import cosine

# 忽略警告用的
import warnings
from bs4 import BeautifulSoup

# 忽略 BeautifulSoup 的特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
# 给wikipedia包的请求加用户头
user_agent = "WDP_lecture_demo/1.0 (s.wang15@student.vu.nl)"
wikipedia.headers = {'User-Agent': user_agent}
# 加载NER（命名实体识别）模型
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
# 初始化 BERT 模型和分词器，用于句子嵌入
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# 计算句子的 BERT 嵌入
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    sentence_embedding = torch.mean(embeddings, dim=1).squeeze()
    return sentence_embedding


# 计算余弦相似度
def cosine_similarity(embedding1, embedding2):
    embedding1_np = embedding1.numpy()
    embedding2_np = embedding2.numpy()
    return 1 - cosine(embedding1_np, embedding2_np)


embedding_cache = {}


def get_cached_sentence_embedding(sentence):
    if sentence in embedding_cache:
        return embedding_cache[sentence]
    embedding = get_sentence_embedding(sentence)
    embedding_cache[sentence] = embedding
    return embedding


page_cache = {}


def get_cached_page(title):
    if title in page_cache:
        return page_cache[title]
    try:
        page = wikipedia.page(title, auto_suggest=True)
        page_cache[title] = page
        return page
    except (wikipedia.PageError, wikipedia.DisambiguationError):
        return None


def get_wikipedia_link(entity, context_sentence, threshold=0.5):
    try:
        # 获取上下文嵌入，并使用缓存
        context_embedding = get_cached_sentence_embedding(context_sentence)

        # 尝试直接获取页面
        try:
            page = wikipedia.page(entity, auto_suggest=False)
        except wikipedia.PageError:
            # 如果找不到页面则使用搜索进行回退
            search_results = wikipedia.search(entity)
            if search_results:
                page = get_cached_page(search_results[0])
                if not page:  # 如果未找到页面
                    return None
            else:
                return None
        except wikipedia.DisambiguationError as e:
            # 处理消歧义页面
            disambiguation_pages = e.options
            best_match = None
            best_similarity = 0

            for option in disambiguation_pages:
                candidate_page = get_cached_page(option)
                if not candidate_page:  # 如果 candidate_page 是 None，跳过
                    continue

                # 缓存候选页面的标题和摘要嵌入
                candidate_title_embedding = get_cached_sentence_embedding(candidate_page.title)
                candidate_summary_embedding = get_cached_sentence_embedding(candidate_page.summary[:500])

                # 计算相似度
                title_similarity = cosine_similarity(context_embedding, candidate_title_embedding) * 0.3
                summary_similarity = cosine_similarity(context_embedding, candidate_summary_embedding) * 0.7
                similarity_score = title_similarity + summary_similarity

                # 更新最佳匹配
                if similarity_score > best_similarity:
                    best_match = candidate_page
                    best_similarity = similarity_score

            if best_match and best_similarity > threshold:
                return best_match.url
            else:
                return None

        # 缓存标题和摘要的嵌入，避免重复计算
        title_embedding = get_cached_sentence_embedding(page.title)
        summary_embedding = get_cached_sentence_embedding(page.summary[:500])

        # 检查页面顶部标注
        page_html = page.html()
        soup = BeautifulSoup(page_html, "html.parser")

        # 处理标注段落中的链接
        hatnote = soup.find("div", {"class": "hatnote"})
        if hatnote:
            links_in_hatnote = [a.get_text() for a in hatnote.find_all("a")]

            # 遍历标注链接中的候选页面
            for link_text in links_in_hatnote:
                candidate_page = get_cached_page(link_text)
                if not candidate_page:  # 如果 candidate_page 是 None，跳过
                    continue

                # 缓存候选页面的标题和摘要嵌入
                candidate_title_embedding = get_cached_sentence_embedding(candidate_page.title)
                candidate_summary_embedding = get_cached_sentence_embedding(candidate_page.summary[:500])

                # 计算相似度
                title_similarity = cosine_similarity(context_embedding, candidate_title_embedding) * 0.3
                summary_similarity = cosine_similarity(context_embedding, candidate_summary_embedding) * 0.7
                similarity_score = title_similarity + summary_similarity

                # 满足相似度要求时直接返回
                if similarity_score > threshold:
                    return candidate_page.url

        # 如果没有找到合适的标注链接，使用原始页面内容计算相似度
        title_similarity = cosine_similarity(context_embedding, title_embedding) * 0.3
        summary_similarity = cosine_similarity(context_embedding, summary_embedding) * 0.7
        similarity_score = title_similarity + summary_similarity

        if similarity_score > threshold:
            return page.url
        else:
            return None

    except Exception as e:
        # 捕获其他异常并记录日志
        print(f"Error while processing entity '{entity}': {e}")
        return None



def extract_entities_with_wiki(text):
    # 使用 NER 模型识别实体
    entities = ner_pipeline(text)
    # 为每个实体查找 Wikipedia 页面链接
    result = {}
    for ent in entities:
        entity_name = ent["word"]
        wiki_url = get_wikipedia_link(entity_name, text)
        if wiki_url:
            result[entity_name] = wiki_url
    return result


# text = (
#     "Princess Zelda is one of Nintendo's best-known characters, but she'd never starred in one of its games until this year."
#     "Despite lending her name to the Legend of Zelda series, she'd always played a supporting role behind regular hero Link."
#     "That all changed with Echoes of Wisdom, released a few weeks ago.")
text = (
    "Is Managua the capital of Nicaragua? ")

entities_with_links = extract_entities_with_wiki(text)
for entity, url in entities_with_links.items():
    print(f"{entity} ⇒ {url}")
