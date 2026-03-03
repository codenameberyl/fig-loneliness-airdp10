import numpy as np
import spacy
from textblob import TextBlob
from src.config import FIRST_PERSON, SOCIAL_WORDS

nlp = spacy.load("en_core_web_sm")


def extract_linguistic_features(texts):
    features = []

    for doc in nlp.pipe(texts, disable=["ner"]):
        tokens = [token.text.lower() for token in doc]

        first_person_ratio = sum(t in FIRST_PERSON for t in tokens) / (len(tokens) + 1)
        social_ratio = sum(t in SOCIAL_WORDS for t in tokens) / (len(tokens) + 1)
        question_marks = doc.text.count("?")
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents])

        sentiment = TextBlob(doc.text).sentiment.polarity

        features.append(
            [
                first_person_ratio,
                social_ratio,
                question_marks,
                avg_sentence_length,
                sentiment,
            ]
        )

    return np.array(features)
