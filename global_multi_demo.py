from transformers import pipeline, BertTokenizer, BertModel
import wikipedia
import torch
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor
import warnings
from bs4 import BeautifulSoup

# 忽略 BeautifulSoup 的特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

# Wikipedia 请求头设置
user_agent = "WDP_lecture_demo/1.0 (s.wang15@student.vu.nl)"
wikipedia.headers = {'User-Agent': user_agent}

# 加载 NER 模型
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 全局线程池
global_thread_pool = ThreadPoolExecutor(max_workers=10)

# 缓存字典
embedding_cache = {}
page_cache = {}
search_cache = {}


# 计算句子的 BERT 嵌入
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return torch.mean(embeddings, dim=1).squeeze()


# 获取或缓存句子的嵌入
def get_cached_sentence_embedding(sentence):
    if sentence in embedding_cache:
        return embedding_cache[sentence]
    embedding = get_sentence_embedding(sentence)
    embedding_cache[sentence] = embedding
    return embedding


# 获取或缓存 Wikipedia 页面
def get_cached_page(title):
    if title in page_cache:
        return page_cache[title]
    try:
        page = wikipedia.page(title, auto_suggest=True)
        page_cache[title] = page
        return page
    except (wikipedia.PageError, wikipedia.DisambiguationError):
        return None


# Wikipedia 搜索并缓存结果
def cached_search(entity):
    if entity in search_cache:
        return search_cache[entity]
    results = wikipedia.search(entity)
    search_cache[entity] = results
    return results


# 计算余弦相似度
def cosine_similarity(embedding1, embedding2):
    embedding1_np = embedding1.numpy()
    embedding2_np = embedding2.numpy()
    return 1 - cosine(embedding1_np, embedding2_np)


# 单个候选页面的处理逻辑
def process_candidate(option, context_embedding):
    try:
        candidate_page = get_cached_page(option)
        if not candidate_page:
            return None, 0

        candidate_title_embedding = get_cached_sentence_embedding(candidate_page.title)
        candidate_summary_embedding = get_cached_sentence_embedding(candidate_page.summary[:500])
        title_similarity = cosine_similarity(context_embedding, candidate_title_embedding) * 0.3
        summary_similarity = cosine_similarity(context_embedding, candidate_summary_embedding) * 0.7
        similarity_score = title_similarity + summary_similarity

        return candidate_page, similarity_score
    except Exception as e:
        print(f"Error processing candidate '{option}': {e}")
        return None, 0


# 获取 Wikipedia 链接
def get_wikipedia_link(entity, context_sentence, context_embedding=None, threshold=0.5):
    try:
        if context_embedding is None:
            context_embedding = get_cached_sentence_embedding(context_sentence)

        # 尝试直接获取页面
        try:
            page = wikipedia.page(entity, auto_suggest=False)
        except wikipedia.PageError:
            search_results = cached_search(entity)
            if search_results:
                page = get_cached_page(search_results[0])
            else:
                return None
        except wikipedia.DisambiguationError as e:
            disambiguation_pages = e.options
            best_match, best_similarity = None, 0

            # 多线程处理消歧义候选页面
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(
                    lambda option: process_candidate(option, context_embedding),
                    disambiguation_pages
                )
                for candidate_page, similarity_score in results:
                    if candidate_page and similarity_score > best_similarity:
                        best_match = candidate_page
                        best_similarity = similarity_score

            return best_match.url if best_match and best_similarity > threshold else None

        # BeautifulSoup 检查页面顶部标注
        page_html = page.html()
        soup = BeautifulSoup(page_html, "html.parser")
        hatnote = soup.find("div", {"class": "hatnote"})
        best_hatnote_match, best_hatnote_similarity = None, 0

        if hatnote:
            # 提取每个链接的文本和 URL
            links_in_hatnote = [(a.get_text(), a['href']) for a in hatnote.find_all("a", href=True)]

            # 多线程处理 hatnote 链接
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(
                    lambda link: process_candidate(link[0], context_embedding),
                    links_in_hatnote
                )
                for candidate_page, similarity_score in results:
                    if candidate_page and similarity_score > best_hatnote_similarity:
                        best_hatnote_match = candidate_page
                        best_hatnote_similarity = similarity_score

        # 检查当前页面相似度
        title_embedding = get_cached_sentence_embedding(page.title)
        summary_embedding = get_cached_sentence_embedding(page.summary[:500])
        title_similarity = cosine_similarity(context_embedding, title_embedding) * 0.3
        summary_similarity = cosine_similarity(context_embedding, summary_embedding) * 0.7
        original_page_similarity = title_similarity + summary_similarity

        # 比较 hatnote 和原始页面的相似度
        if best_hatnote_match and best_hatnote_similarity > original_page_similarity:
            return best_hatnote_match.url

        # 如果没有更好的 hatnote 页面，返回原始页面
        if original_page_similarity > threshold:
            return page.url

        return None

    except Exception as e:
        print(f"Error processing entity '{entity}': {e}")
        return None



# 提取单个实体的 Wikipedia 链接
def process_entity(entity, context_sentence, context_embedding):
    try:
        wiki_url = get_wikipedia_link(entity, context_sentence, context_embedding)
        return entity, wiki_url
    except Exception as e:
        print(f"Error processing entity '{entity}': {e}")
        return entity, None


# 全局多线程处理多个实体
def extract_entities_with_wiki_multithread(text):
    entities = ner_pipeline(text)
    context_embedding = get_cached_sentence_embedding(text)

    futures = [
        global_thread_pool.submit(process_entity, ent["word"], text, context_embedding)
        for ent in entities
    ]

    result = {}
    for future in futures:
        entity, wiki_url = future.result()
        if wiki_url:
            result[entity] = wiki_url

    return result


# 示例文本
text = (
    "Princess Zelda is one of Nintendo's best-known characters, but she'd never starred in one of its games until this year. "
    "Despite lending her name to the Legend of Zelda series, she'd always played a supporting role behind regular hero Link. "
    "That all changed with Echoes of Wisdom, released a few weeks ago."
)
# text = (
#     "Apple Inc. is a world-renowned high-tech company"
# )

# 提取实体并输出链接
entities_with_links = extract_entities_with_wiki_multithread(text)
for entity, url in entities_with_links.items():
    print(f"{entity} ⇒ {url}")
