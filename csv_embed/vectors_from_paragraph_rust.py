import openai
import csv
import sys
import os
import re

API_BASE_URL = "https://llama8b.gaia.domains/v1"
MODEL_NAME = "llama"
API_KEY = "gaia-NjhkMGM2NmUtMzg5Ny00YzFkLWI4YWQtYzMwMTA2MTQ0ZjE3-aCmbMEJizuL70Xu5"

def summarize(source_text):
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a teacher of the Rust programming language. Respond with a comprehensive summary of the text in the user message. Cover conceptual topics, techniques and conclusions in your summary.",
            },
            {
                "role": "user",
                "content": source_text,
            }
        ],
        model=MODEL_NAME,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def qgen(source_text):
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a teacher of the Rust programming language. Respond with a list of 10 questions. Each question must be about a concept or technique in Rust programming. The text in the user message must contain specific answers to each question. The question should not reference lines or variables in code listings in the user message. Each question must be on its own line. Just list the questions without any introductory text or numbers.",
            },
            {
                "role": "user",
                "content": source_text,
            }
        ],
        model=MODEL_NAME,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def agen(source_text, question):
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Give a comprehensive and well-reasoned answer to the user question about the Rust programming language strictly based on the context below.\n" + source_text,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model=MODEL_NAME,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def main():
    results = [];
    arguments = sys.argv[1:]

    with open(arguments[0], 'r', newline='') as txtfile:
        txt_data = txtfile.read()
        paras = re.split('^\s*$', txt_data, flags=re.MULTILINE)

        current_line = 0
        for p in paras:
            current_line = current_line + 1
            print(str(current_line) + ' of ' + str(len(paras)))
            print(p + "\n\n\n")
            if len(p.strip()) == 0:
                continue

            result = []
            result.append(p)
            result.append(summarize(p))
            results.append(result)

            qs = qgen(p)
            for q in qs.splitlines():
                if len(q.strip()) == 0 or (not q.endswith("?")):
                    continue
                print('question: ' + q)
                result = []
                result.append(p)
                result.append(q + "\n" + agen(p, q))
                results.append(result)

    with open(arguments[1], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()
