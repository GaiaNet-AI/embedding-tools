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
  paragraph_embeddings.wasm embedding chemistry_book 384 chemistry.txt
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

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
   paragraph_embeddings.wasm embedding chemistry_book 384 chemistry.txt -s 5 -m 1024
```

