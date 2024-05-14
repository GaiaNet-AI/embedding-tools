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

## Build the wasm app

```
RUSTFLAGS="--cfg wasmedge --cfg tokio_unstable" cargo build --target wasm32-wasi --release

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

Additionally, you can use the following optional arguments:

- `-m` or `--maximum_context_length`: Specify a context length to truncate and warn for each text segment that exceeds the specified length.
- `-s` or `--start_vector_id`: Specify the starting vector ID, which allows you to run the application multiple times on different documents within the same vector collection.
- `-l` or `--heading_level`: Specify the markdown heading level to chunk the text on markdown documents.

Example usage with optional arguments:

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
  markdown_embed.wasm embedding my_book 384 input.md --heading_level 2 -s 5 -m 1024 -l 2
```

You can now query or snapshot the `my_book` vector collection on the local Qdrant server.

```
curl 'http://localhost:6333/collections/my_book'
```

