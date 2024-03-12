#from corpus_of_documents import corpus_of_documents
#from apikey import apikey
import openai
from langsmith import Client
import phoenix as px



openai.api_key = "aparna-api-key" #change openai key
corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters."
]

def jaccard_similarity(query, document):
    query = query.lower().split(" ")
    document = document.lower().split(" ")
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def return_response(query, corpus_of_documents):
    similarities = []
    for doc in corpus_of_documents:
        similarity = jaccard_similarity(query, doc)
        similarities.append(similarity)
    return corpus_of_documents[similarities.index(max(similarities))]

def return_response_with_api(query, corpus):
    prompt = f"Query: {query}\nCorpus:\n"
    for doc in corpus:
        prompt += f"- {doc}\n"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.5,
        max_tokens=100
    )
    return response.choices[0].message


user_query = "What are some outdoor activities?"
response = return_response_with_api(user_query, corpus_of_documents)
print(response)

