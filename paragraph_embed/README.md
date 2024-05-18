# Plain text paragraghs to embeddings 

This repository is adapted from the [YuanTony/chemistry-assistant](https://github.com/YuanTony/chemistry-assistant/tree/main/rag-embeddings) repository. It chunks a text file based on blank lines, while NOT breaking up code listings, turns each
chunk into a vector embedding, and then saves the vector embedding into a Qdrant database.

## Prerequisites

Please follow the instructions from the original repository to set up the following:

- [Start the Qdrant vector database](https://qdrant.tech/documentation/quick-start/) at `localhost:6333`
- [Download an embedding model](https://huggingface.co/gaianet/all-MiniLM-L6-v2-ggml-model-GGUF/blob/main/all-MiniLM-L6-v2-ggml-model-f16.gguf)

[Install WasmEdge with GGML plugin](https://wasmedge.org/docs/start/install): 

```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-ggml
source /home/azureuser/.bashrc
```

Create a vector collection called `chemistry_book`. Here we create a vector collection with 384 dimensions. You will need to adjust this if you use a different embedding model.

```
curl -X PUT 'http://localhost:6333/collections/chemistry_book' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "vectors": {
      "size": 384,
      "distance": "Cosine",
      "on_disk": true
    }
  }'
```

## Build the wasm app

```
RUSTFLAGS="--cfg wasmedge --cfg tokio_unstable" cargo build --target wasm32-wasi --release

cp target/wasm32-wasi/release/paragraph_embed.wasm .
```

## Generate embeddings

Now, we can run the Wasm app to generate embeddings from a text file [chemistry.txt](chemistry.txt) and save to the Qdrant `chemistry_book` collection.

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
  paragraph_embed.wasm embedding chemistry_book 384 chemistry.txt
```

After it is done, you can check the vectors and their associated payloads by their IDs. THe example below returns the first and last vectors in the `chemistry_book` collection.

```
curl 'http://localhost:6333/collections/chemistry_book/points' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "ids": [0, 1231],
    "with_payload": true,
    "with_vector": false
  }'
```

You can also pass the following options to the program.

* Using `-m` or `--maximum_context_length` to specify a context length in the CLI argument. That is to truncate and warn for each text segment that goes above the context length.
* Using `-s` or `--start_vector_id` to specify the start vector ID in the CLI argument. This will allow us to run this app multiple times on multiple documents on the same vector collection.
* Using `-c` or `--ctx_size` to specify the context size of the input. This defaults to 512.

Example: Use the `all-MiniLM-L6-v2-ggml-model-f16` model, which has a context length of 256 and vector size of 384.

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
   paragraph_embed.wasm embedding chemistry_book 384 chemistry.txt
```

Example: Use the `all-MiniLM-L6-v2-ggml-model-f16` model but start at index 5 and truncate input to 1024 chars to avoid context length overflow.

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
   paragraph_embed.wasm embedding chemistry_book 384 chemistry.txt -s 5 -m 1024
```

Example: use the `nomic-embed-text-v1.5.f16` model, which has a context length of 8192 and vector size of 768. Note that your `chemistry_book` vector collection must be set up to be 768 dimensions.

```
curl -LO https://huggingface.co/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf

wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
   paragraph_embed.wasm embedding chemistry_book 768 chemistry.txt -c 8192
```

## See the vector collection in Qdrant

```
curl http://localhost:6333/collections/chemistry_book
```

Create a snapshot for sharing

```
curl -X POST 'http://localhost:6333/collections/chemistry_book/snapshots'
```

You can find the snapshot file in Qdrant's snapshot directory.
