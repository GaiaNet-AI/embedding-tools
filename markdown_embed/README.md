# Markdown to embeddings 

This tool chunks an input markdown file based on the Markdown heading levels, and then creates a vector embedding for each chunk in a vector database. Additionally, for each non-top-level chunk, the preceding heading levels will be prepended to the chunk.

## Prerequisites

Please follow the instructions from the original repository to set up the following:

- [Start the Qdrant vector database](https://qdrant.tech/documentation/quick-start/) at `localhost:6333`
- [Download an embedding model](https://huggingface.co/gaianet/all-MiniLM-L6-v2-ggml-model-GGUF/blob/main/all-MiniLM-L6-v2-ggml-model-f16.gguf)

[Install WasmEdge with GGML plugin](https://wasmedge.org/docs/start/install):

```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-ggml
source /home/azureuser/.bashrc
```

Create a vector collection called `my_book`. Here we create a vector collection with 384 dimensions. You will need to adjust this if you use a different embedding model.

```
curl -X PUT 'http://localhost:6333/collections/my_book' \
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
cargo build --target wasm32-wasi --release

cp target/wasm32-wasi/release/markdown_embed.wasm .
```

## Generate Embeddings with Heading-Level Chunking

To generate embeddings with heading-level chunking, run the following command:

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
  markdown_embed.wasm embedding collection_name 384 input.md --heading_level N
```

Replace `collection_name` with the name of your Qdrant collection, `input.md` with the path to your input markdown file, and `N` with the desired heading level for chunking (e.g., `2` for chunking at the second heading level).

Additionally, you can use the `-c` or `--ctx_size` parameters to specify the context size of the input. This defaults to 512.

Example: use the `nomic-embed-text-v1.5.f16` model, which has a context length of 8192 and vector size of 768. Note that your `my_book` vector collection must be set up to be 768 dimensions.

```
curl -LO https://huggingface.co/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf

wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
   markdown_embed.wasm embedding my_book 768 input.md -c 8192
```

You can now query or snapshot the `my_book` vector collection on the local Qdrant server.

```
curl 'http://localhost:6333/collections/my_book'
```

