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
RUSTFLAGS="--cfg wasmedge --cfg tokio_unstable" cargo build --target wasm32-wasi --release

cp target/wasm32-wasi/release/csv_embed.wasm .
```

## Generate Embeddings

To generate embeddings, run the following command:

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
  csv_embed.wasm embedding my_book 768 input.csv
```

Additionally, you can use the following optional arguments:

- `-c` or `--ctx_size` to specify the context size of the input. This defaults to 512.
- `-s` or `--start_vector_id`: Specify the starting vector ID, which allows you to run the application multiple times on different documents within the same vector collection.

You can now query or snapshot the `my_book` vector collection on the local Qdrant server.

```
curl 'http://localhost:6333/collections/my_book'
```

You can always delete the vector collection and start over.

```
curl -X DELETE 'http://localhost:6333/collections/my_book'
```

