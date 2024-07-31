# CSV to embeddings 

This tool chunks an input markdown file based on the Markdown heading levels, and then creates a vector embedding for each chunk in a vector database. Additionally, for each non-top-level chunk, the preceding heading levels will be prepended to the chunk.

## Prerequisites

Please follow the instructions from the original repository to set up the following:

[Start the Qdrant vector database](https://qdrant.tech/documentation/quick-start/) at `localhost:6333`.

[Install WasmEdge with GGML plugin](https://wasmedge.org/docs/start/install) and download the embedding model
as follows.

```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s

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
RUSTFLAGS="--cfg wasmedge --cfg tokio_unstable" cargo build --target wasm32-wasi --release

cp target/wasm32-wasi/release/csv_embed.wasm .
```

## Generate Embeddings

To generate embeddings, run the following command:

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
  csv_embed.wasm embedding collection_name 768 input.csv
```

Replace `collection_name` with the name of your Qdrant collection, `input.csv` with the path to your input CSV file.

Additionally, you can use the following optional arguments:

- `-c` or `--ctx_size` to specify the context size of the input. This defaults to 512.
- `-s` or `--start_vector_id`: Specify the starting vector ID, which allows you to run the application multiple times on different documents within the same vector collection.

You can now query or snapshot the `my_book` vector collection on the local Qdrant server.

```
curl 'http://localhost:6333/collections/my_book'
```

