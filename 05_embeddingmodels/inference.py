# Warning control
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, \
                         DPRContextEncoder, DPRQuestionEncoder

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def cosine_similarity_matrix(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / norms
    similarity_matrix = np.inner(normalized_features, normalized_features)
    rounded_similarity_matrix = np.round(similarity_matrix, 4)
    return rounded_similarity_matrix

answers = [
    "What is the tallest mountain in the world?",
    "The tallest mountain in the world is Mount Everest.",
    "Mount Shasta",
    "I like my hike in the mountains",
    "I am going to a yoga class"
]

question = 'What is the tallest mountain in the world?'

answer_tokenizer = AutoTokenizer \
                   .from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
answer_encoder = DPRContextEncoder \
                   .from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

question_tokenizer = AutoTokenizer \
                   .from_pretrained("facebook/dpr-question_encoder-multiset-base")
question_encoder = DPRQuestionEncoder \
                   .from_pretrained("facebook/dpr-question_encoder-multiset-base")


# Compute the question embeddings
question_tokens = question_tokenizer(question, return_tensors="pt")["input_ids"]
question_embedding = question_encoder(question_tokens).pooler_output.flatten().tolist()
print(question_embedding[:10], len(question_embedding))

sim = []
for answer in answers:
    answer_tokens = answer_tokenizer(answer, return_tensors="pt")["input_ids"]
    answer_embedding = answer_encoder(answer_tokens).pooler_output.flatten().tolist() 
    sim.append(cosine_similarity_matrix(np.stack([question_embedding, answer_embedding]))[0,1])

print(sim)
best_inx = np.argmax(sim)
print(f"Question = {question}")
print(f"Best answer = {answers[best_inx]}")