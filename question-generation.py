import json
import os
from openai import OpenAI

client = OpenAI(api_key="open-ai-key")
# --- Load topic keywords ---
with open("combined_cluster_topic_keywords.json", "r") as f:
    topic_data = json.load(f)
x=0
for cluster_id, topics in topic_data.items():
    if(x==1):
        break
    #x=x+1
    for topic_id, keyword_types in topics.items():
        lda_keywords = keyword_types.get("lda", [])
        prompt = f""
        # --- Call OpenAI API ---
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating valid questions."},
            {"role": "user", "content": prompt}
        ]
        )
        print(response.choices[0].message.content)
        with open("gpt_questions.txt", "a") as out_file:
            out_file.write("\n")
            out_file.write(response.choices[0].message.content)
print(" Done processing ")
