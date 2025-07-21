import os
import pickle
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from tavily import TavilyClient

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from prepare_index import preprocess_symptoms


load_dotenv()
client = TavilyClient(os.getenv("TAVILY_API_KEY"))


def search_disease_infomation_tavily(query: str, top_k: int = 5) -> str:
    """
    Search for disease information using Tavily API based on the provided query.
    Args:
        query (str): The query string containing symptoms or disease information.
        top_k (int, optional): The number of top results to return. Defaults to 3.
    Returns:
        str: Formatted string containing Vietnamese medical information related to the query, separated by "---" delimiters.
    """
    response = client.qna_search(
        query=query,
        max_results=top_k,
        country="vietnam",
    )
    # results = response["results"]
    # formatted_result = f"Kết quả tìm kiếm từ web:\n" + "\n---\n".join(
    #     [f"Nguồn: {res['title']}\nNội dung: {res['content']}" for res in results]
    # )

    return response


class VectorizerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == preprocess_symptoms.__name__:
            return preprocess_symptoms
        # if name == vietnamese_medical_stopwords:
        #     return vietnamese_medical_stopwords
        return super().find_class(module, name)


def load_vectorizer(path):
    with open(path, "rb") as file:
        return VectorizerUnpickler(file).load()


# load data
data = pd.read_csv("data/symptoms.csv")

# Load the tfidf_index and tfidf_vectorizer from disk
tfidf_index = sparse.load_npz("data/tfidf_index.npz")
tfidf_vectorizer = load_vectorizer("data/tfidf_vectorizer.pkl")


def search_disease_infomation(query_symptoms: str, top_k: int = 3) -> str:
    """
    Search for diseases based on the given query symptoms using TF-IDF similarity.

    Args:
      query_symptoms (str): The symptoms to search for (as a string).
      top_k (int, optional): The number of top results to return. Defaults to 3.

    Returns:
      str: Formatted string containing disease information with Vietnamese labels, separated by "---" delimiters. Each entry includes disease name and symptoms.
    """
    query_vector = tfidf_vectorizer.transform([query_symptoms])
    similarities = cosine_similarity(query_vector, tfidf_index).flatten()
    # Filter indices where similarity is greater than 0
    relevant_indices = np.where(similarities > 0)[0]

    # If no relevant matches found, return empty string
    if len(relevant_indices) == 0:
        return "Không tìm thấy bệnh nào phù hợp với triệu chứng đã nhập."

    # Get similarities for relevant indices only
    relevant_similarities = similarities[relevant_indices]

    # Sort relevant indices by similarity in descending order
    sorted_relevant_indices = relevant_indices[relevant_similarities.argsort()[::-1]]

    # Select top_k from the relevant indices
    top_k = min(top_k, len(sorted_relevant_indices))
    top_indices = sorted_relevant_indices[:top_k]

    results = [
        {
            "disease": data.at[idx, "Disease"],
            "symptoms": data.at[idx, "symptoms"],
        }
        for idx in top_indices
    ]

    # Format the results for llm context engineering
    formatted_results = "\n---\n".join("Bệnh " + item["symptoms"] for item in results)

    return formatted_results
