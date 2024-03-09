from gensim import corpora, models, similarities
from pprint import pprint

def create_vsm_model(document_list: list):
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    texts = []
    # 导入停用词列表
    for document in document_list:
        text = []
        for word in document.lower().split():
            if not stop_words.__contains__(word):
                text.append(word)
        texts.append(text)
    # 分词和建立文档-词频矩阵
    texts = [[word for word in document.lower().split()] for document in document_list]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 训练VSM模型
    tfidf_model = models.TfidfModel(corpus, id2word=dictionary, normalize=True)
    tfidf_corpus = tfidf_model[corpus]

    # 创建一个用于计算相似度的索引
    index = similarities.MatrixSimilarity(tfidf_corpus)
    return dictionary, tfidf_model, index


def calculate_sim_by_vsm(dictionary, model, index, query_document):
    # 将查询文档转换为VSM空间
    query_bow = dictionary.doc2bow(query_document.lower().split())
    query_vsm = model[query_bow]

    # 计算相似度
    sims = index[query_vsm]

    # 打印相似度结果

    return sims

# 示例文本数据
documents = [
    "This is a sample document.",
    "Another document example.",
    "Yet another document.",
    "One more example document.",
]
d, m, i = create_vsm_model(documents)
calculate_sim_by_vsm(d, m, i, "sample document.")

