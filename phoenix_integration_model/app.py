import json
import pandas as pd
import phoenix as px
import requests


url = "https://raw.githubusercontent.com/anthropics/evals/main/persona/anti-LGBTQ-rights.jsonl"

response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Split the content by lines
    lines = response.text.splitlines()

    # Parse each line as a JSON object and create a list of dictionaries
    data = [json.loads(line) for line in lines]

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)
else:
    print(f"Error: {response.status_code}")


from arize.pandas.embeddings import EmbeddingGenerator, UseCases

df = df.reset_index(drop=True)

generator = EmbeddingGenerator.from_use_case(
    use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
    model_name="distilbert-base-uncased",
    tokenizer_max_length=512,
    batch_size=100,
)
df["statement_vector"] = generator.generate_embeddings(text_col=df["statement"])
df["question_vector"] = generator.generate_embeddings(text_col=df["question"])

schema = px.Schema(
    response_column_names=px.EmbeddingColumnNames(
        vector_column_name="statement_vector", raw_data_column_name="statement"
    ),
    prompt_column_names=px.EmbeddingColumnNames(
        vector_column_name="question_vector", raw_data_column_name="question"
    ),
    prediction_label_column_name="answer_matching_behavior",
)

px.launch_app(px.Dataset(df, schema))
