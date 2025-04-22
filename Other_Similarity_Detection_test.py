import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from simphile import jaccard_similarity, euclidian_similarity, compression_similarity
import Levenshtein
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

original_text_path = os.path.join(current_dir, 'Example of PlagiarismCheck', 'Originaltext.json')
ai_text_data_path = os.path.join(current_dir, 'Example of PlagiarismCheck', 'AItext.json')

with open(original_text_path, 'r', encoding='utf-8') as file:
    original_text_data = json.load(file)

with open(ai_text_data_path, 'r', encoding='utf-8') as file:
    ai_text_data = json.load(file)

vectorizer = TfidfVectorizer()

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def lcs_similarity(str1, str2):
    lcs_length = lcs(str1, str2)
    max_length = max(len(str1), len(str2))
    if max_length == 0:
        return 0.0
    return lcs_length / max_length

all_similarity_results = {}

for idx, original_text in enumerate(original_text_data['Original text']):
    original_text_content = original_text['text']
    original_text_tfidf_matrix = vectorizer.fit_transform([original_text_content])

    ai_text_key = f'Original text{idx + 1}'
    if ai_text_key in ai_text_data:
        ai_texts = ai_text_data[ai_text_key]
        for i, ai_text in enumerate(ai_texts):
            ai_text_content = ai_text.get('text', '')
            ai_text_id = f'answerresult{i + 1}'

            if ai_text_content:
                ai_text_tfidf_matrix = vectorizer.transform([ai_text_content])

                cosine_similarities = cosine_similarity(ai_text_tfidf_matrix, original_text_tfidf_matrix)
                cosine_similarity_score = cosine_similarities[0][0]

                vector_a = original_text_tfidf_matrix.toarray().flatten()
                vector_b = ai_text_tfidf_matrix.toarray().flatten()

                lcs_score = lcs_similarity(original_text_content, ai_text_content)

                levenshtein_distance = Levenshtein.distance(original_text_content, ai_text_content)

                levenshtein_similarity_score = 1 - levenshtein_distance / max(len(original_text_content), len(ai_text_content))

                jaccard_score = jaccard_similarity(original_text_content, ai_text_content)

                compression_score = compression_similarity(original_text_content, ai_text_content)

                similarity_results = {
                    'Original Text ID': original_text['id'],
                    'AI Text ID': ai_text_id,
                    'LCS Similarity Score': lcs_score,
                    'Levenshtein Similarity Score': levenshtein_similarity_score,
                    'Jaccard Similarity Score': jaccard_score,
                    'Compression Similarity Score': compression_score,
                    'Cosine Similarity Score': cosine_similarity_score,
                    'Euclidean Similarity Score': euclidian_similarity(original_text_content, ai_text_content)
                }

                all_similarity_results[f'{original_text["id"]}_{ai_text_id}'] = similarity_results
    else:
        print(f"No data found for {ai_text_key} in AItext.json")

result_df = pd.DataFrame(all_similarity_results).T
result_excel_path = os.path.join(current_dir, 'textdata', 'SimilarityResults.xlsx')
result_df.to_excel(result_excel_path, index_label='Text Pair')