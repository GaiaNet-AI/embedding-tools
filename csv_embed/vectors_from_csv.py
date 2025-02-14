import openai
import csv
import sys
import os

API_BASE_URL = "https://llama8b.gaia.domains/v1"
MODEL_NAME = "llama"
API_KEY = "gaia-NjhkMGM2NmUtMzg5Ny00YzFkLWI4YWQtYzMwMTA2MTQ0ZjE3-aCmbMEJizuL70Xu5"

def summarize(source_text):
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Respond with a comprehensive summary of the text in the user message",
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
                "content": "Respond with a list of 10 questions. The text in the user message must contain specific answers to each question. Each question must be complete without references to unclear context such as \"this team\" or \"that lab\". Each question must be on its own line. Just list the questions without any introductory text or numbers.",
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
                "content": "Give a comprehensive and well-reasoned answer to the user question strictly based on the context below.\n" + source_text,
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

    with open(arguments[0], 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            print('row start')

            result = []
            result.append(row[0])
            result.append(summarize(row[0]))
            results.append(result)

            qs = qgen(row[0])
            for q in qs.splitlines():
                if len(q.strip()) == 0:
                    continue
                print('question: ' + q)
                result = []
                result.append(row[0])
                result.append(q + "\n" + agen(row[0], q))
                results.append(result)

            print('row end')

    with open(arguments[1], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()
