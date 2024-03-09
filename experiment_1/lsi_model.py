from gensim import corpora, models, similarities
from pprint import pprint

# 示例文本数据
documents = [
    "This is a sample document.",
    "Another document example.",
    "Yet another document.",
    "One more example document.",
]
def create_lsi_model(document_list,num_topics):
    """
        document_list: 语料库文档
    """
    # 分词和建立文档-词频矩阵
    texts = [[word for word in document.lower().split()] for document in document_list]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 训练LSI模型
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

    # 创建一个用于计算相似度的索引
    index = similarities.MatrixSimilarity(lsi_model[corpus])
    return dictionary, lsi_model, index


# 使用该方法前，需要使用create_model获取参数
def calculate_sim_by_lsi(dictionary, lsi_model, index, query_document):
    # 将查询文档转换为LSI空间
    query_bow = dictionary.doc2bow(query_document.lower().split())
    query_lsi = lsi_model[query_bow]

    # 计算相似度
    sims = index[query_lsi]

    # 打印相似度结果
    # print("Similarity Scores:")
    # for i, score in enumerate(sims):
    #     print(f"Document {i + 1}: {score}")

    return sims

# d,m,i = create_lsi_model(documents)
# calculate_sim_by_lsi(d, m, i, "This is a sample document.")