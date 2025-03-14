# CSV to embeddings 

The CSV file has two columns. The first is full text column for the source text of a chunk or a chapter.
The second is a summary of the full text. This tool turns the summary text into vectors, and store
each vector with the associated full text. That allows queries to be matched to the more concise summary
but have the full text in the RAG context.

## Prerequisites

Please follow the instructions from the original repository to set up the following:

[Start the Qdrant vector database](https://qdrant.tech/documentation/quick-start/) at `localhost:6333`.

[Install WasmEdge with GGML plugin](https://wasmedge.org/docs/start/install).

```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s
```

Download the embedding model.

```
curl -LO https://huggingface.co/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf
```

Create a vector collection called `my_book`. Here we create a vector collection with 768 dimensions. You will need to adjust this if you use a different embedding model.

```
curl -X PUT 'http://localhost:6333/collections/my_book' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "vectors": {
      "size": 768,
      "distance": "Cosine",
      "on_disk": true
    }
  }'
```

## Build the wasm app

```
cargo build --target wasm32-wasip1 --release

cp target/wasm32-wasi/release/csv_embed.wasm .
```

## Generate Embeddings

To generate embeddings, run the following command:

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
  csv_embed.wasm embedding my_book 768 input.csv -c 8192
```

You can now query or snapshot the `my_book` vector collection on the local Qdrant server.

```
curl 'http://localhost:6333/collections/my_book'
```

Create a snapshot for sharing

```
curl -X POST 'http://localhost:6333/collections/my_book/snapshots'
```

You can always delete the vector collection and start over.

```
curl -X DELETE 'http://localhost:6333/collections/my_book'
```

## Create a vectors file from a single column CSV file

We start from a single column CSV file. Each row has only one cell, and it contains a complete article or chapter. Let's call it `source_text.csv`.

Run a Python script to add a second column and create a new CSV file `vectors.csv`. The second column contains the summarization and QnA of the source text. It is then converted into vectors using the above `csv_embed.wasm` utility.

```
python3 vectors_from_csv.py source_text.csv vectors.csv
```

The script uses the `llama.us.gaianet.network` node to summarize and generate QnA pairs based on the source text. You can switch to a different LLM (e.g., GPT-4) by changing the configuration at the top of the Python file.

Example

```
curl -LO https://huggingface.co/datasets/gaianet/samsung/resolve/main/samsung-galaxy-s24-source.csv
python3 vectors_from_csv.py samsung-galaxy-s24-source.csv samsung-galaxy-s24-pairs.csv


wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
  csv_embed.wasm embedding samsung_galaxy_s24 768 samsung-galaxy-s24-pairs.csv -c 8192
```

## Create a vectors file from a text file

We start from a text file. Each paragraph, separated by a blank line, is an article or a chapter. Let's call it `source.txt`.

Run a Python script to create a new CSV file `vectors.csv`. The first column is the source paragraph text and the second column contains the summarization and QnA of the source text. It is then converted into vectors using the above `csv_embed.wasm` utility.

```
python3 vectors_from_paragraph.py source.txt vectors.csv
```

The script uses the `llama.us.gaianet.network` node to summarize and generate QnA pairs based on the source text. You can switch to a different LLM (e.g., GPT-4) by changing the configuration at the top of the Python file.

Example

```
curl -LO https://huggingface.co/datasets/gaianet/learn-rust/resolve/main/rust-books-source.txt
python3 vectors_from_paragraph.py rust-books-source.txt rust-books-pairs.csv


wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
  csv_embed.wasm embedding rust_books 768 rust-books-pairs.csv -c 8192
```


