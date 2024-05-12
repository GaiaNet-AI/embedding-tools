# Embedding-tools
This repository is a modification of the original [YuanTony/chemistry-assistant](https://github.com/YuanTony/chemistry-assistant/tree/main/rag-embeddings) repository. The only difference is that instead of using blank lines as the basis for chunking the input text, a `heading_level` parameter has been added to chunk the text based on the Markdown heading levels. Additionally, for each non-top-level chunk, the preceding heading levels will be prepended to the chunk.

## Prerequisites

Please follow the instructions from the original repository to set up the following:

- Start the Qdrant vector database
- Install WasmEdge with GGML plugin
- Build the program to generate embeddings
- Download an embedding model

## Generate Embeddings with Heading-Level Chunking

To generate embeddings with heading-level chunking, run the following command:

```
cp target/wasm32-wasi/release/create_embeddings.wasm .
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
  create_embeddings.wasm embedding collection_name 384 input_file --heading_level N
```

Replace `collection_name` with the name of your Qdrant collection, `input_file` with the path to your input file, and `N` with the desired heading level for chunking (e.g., `2` for chunking at the second heading level).

Additionally, you can use the following optional arguments:

- `-m` or `--maximum_context_length`: Specify a context length to truncate and warn for each text segment that exceeds the specified length.
- `-s` or `--start_vector_id`: Specify the starting vector ID, which allows you to run the application multiple times on different documents within the same vector collection.
- `-l` or `--heading_level`: Specify the markdown heading level to chunk the text on markdown documents.

Example usage with optional arguments:

```
wasmedge --dir .:. \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
  create_embeddings.wasm embedding chemistry_book 384 input.txt --heading_level 2 -s 5 -m 1024 -l 2
```