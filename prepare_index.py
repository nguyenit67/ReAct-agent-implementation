# %%
import json
import re
import pickle
import unicodedata
import pandas as pd

from scipy import sparse
from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# apply tf-idf filtering on Question column

# --- 1. Pre-processing ---
# Define a list of Vietnamese stop words to ignore common words
# We include words from the repetitive question "Tôi có thể đang bị bệnh gì?"
vietnamese_medical_stopwords = set(
    (
        "tôi",
        "bị",
        "là",
        # "có",
        "của",
        # "và",
        "khi",
        "thì",
        "mà",
        # "ở",
        "đang",
        "cảm_thấy",
        "hiện",
        "tại",
        "các",
        "triệu_chứng",
        "như",
        "có_thể",
        "bệnh",
        "gì",
    )
)


def preprocess_symptoms(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)  # Normalize unicode characters
    text = text_normalize(text.lower())
    tokens = word_tokenize(text, format="text").split()
    tokens = [token for token in tokens if re.match(r"^[^:,;]+$", token) and token not in vietnamese_medical_stopwords]
    return tokens


# %%

if __name__ == "__main__":

    # %%
    # data = pd.read_csv("data/ViMedical_Disease.csv")
    # data.groupby("Disease").agg("sum").reset_index().to_csv("data/ViMedical_Disease_summary.csv", index=False)

    # %%

    # tfidf_vocabulary = set(tfidf_vectorizer.get_feature_names_out())

    def get_symptom_keywords(text: str) -> str:
        keywords = preprocess_symptoms(text)
        # keywords = [token for token in tokens if token in tfidf_vocabulary]
        result = " ".join(keywords)
        result = re.sub(r"[,;]", ",", result)
        result = re.sub(r"[.?!]", ";", result)
        # Remove extra spaces
        result = re.sub(r"(\s,)+", ",", result)
        result = re.sub(r"(\s;)+", ";", result)
        result = re.sub(r"(\s:)+", ":", result)
        result = re.sub(r"_", " ", result)
        if result.endswith(";"):
            result = result[:-1]
        return result

    with open("data/intent_train.json", "r", encoding="utf-8") as file:
        intent_data = json.load(file)

    more_data = []
    disease_map = {
        "emotional problems": "vấn đề về sức khỏe tinh thần và cảm xúc",
        "common cold": "cảm lạnh thông thường",
        "cardiovascular": "bệnh tim mạch",
        "blood pressure": "cao huyết áp",
    }

    for intent in intent_data["intents"]:
        tag = intent.get("tag")
        if tag in ("greeting", "problem_solving", "booking"):
            continue

        symptoms = ", ".join(intent.get("patterns", []))
        if symptoms:
            # symptoms = get_symptom_keywords(symptoms)
            disease = disease_map.get(tag, tag).capitalize()
            more_data.append(
                {
                    "Disease": disease,
                    "symptoms": disease + ": " + symptoms.lower(),
                }
            )

    # Add the new symptoms to the data_symptoms DataFrame
    # data = pd.read_csv("data/ViMedical_Disease_summary.csv")

    # data["Question"] = data["Disease"] + ": " + data["Question"]
    # data["symptoms"] = data["Question"].apply(get_symptom_keywords)

    # data_symptoms = data[["Disease", "symptoms"]]
    # more_data_df = pd.DataFrame(more_data)
    # data_symptoms = pd.concat([data_symptoms, more_data_df], ignore_index=True)
    data_symptoms = pd.DataFrame(more_data)

    data_symptoms.to_csv("data/symptoms.csv", index=False)

    # Add symptoms to the intent data
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=preprocess_symptoms,
        # stop_words=vietnamese_medical_stopwords,
        ngram_range=(1, 4),  # Unigrams and bigrams
    )

    tfidf_index = tfidf_vectorizer.fit_transform(data_symptoms["symptoms"])

    # --- 2. Save the TF-IDF index and vectorizer ---

    # save the tfidf_index and tfidf_vectorizer to disk
    sparse.save_npz("data/tfidf_index.npz", tfidf_index)
    # joblib.dump(tfidf_vectorizer, "data/tfidf_vectorizer.pkl")

    with open("data/tfidf_vectorizer.pkl", "wb") as pkl_file:
        pickle.dump(tfidf_vectorizer, pkl_file)

# %%
