## Prompt Caching - 

Based on @sps response, my understanding is as follows:

When we have a combined input of (System Prompt + user input A), and the total token count reaches or exceeds 1024 tokens, the server-side will begin to cache any components that remain consistent throughout the conversation.

This assumes that user A maintains a consistent dialogue, meaning that the system prompt is sent with every new input, as illustrated below:

Example of Inputs:

SEND: System Prompt + userA_input(1)
RESPONSE: gptResponse(1)

message_history = System Prompt + userA_input(1) + gptResponse(1)

SEND: System Prompt + userA_input(1) + gptResponse(1) + userA_input(2)
RESPONSE: gptResponse(2)

message_history = System Prompt + userA_input(1) + gptResponse(1) + userA_input(2) + gptResponse(2)

As the conversation history grows, the inputs become increasingly lengthy. It is reasonable to conclude that the server will analyse the repeated elements within this input and begin caching them. Therefore, as the system prompt and prior user inputs are repeated in subsequent messages, the server-side caching is likely to minimize processing costs for users.

## KV - Caching - 

. Key-Value (KV) Cache

When generating text token by token, recomputing the keys and values for all previous tokens at each step is inefficient. The KV cache addresses this inefficiency by storing the keys and values computed for each token, allowing them to be reused in subsequent steps.

Purpose of the KV Cache:

Avoid Redundant Computations: By caching keys and values, the model doesn’t need to recompute them for previous tokens when generating each new token.
How It Works:

Initial Token:

Compute the queries, keys, and values for the first token.
Store the keys and values in the KV cache.
Subsequent Tokens:

For each new token, compute its query, key, and value.
Append the new key and value to the KV cache.
Use the query of the current token and the keys and values from the cache to compute attention outputs.
Implementation Details:

KV Cache Structure:

Typically stored as tensors with dimensions:

Keys: [Number of Layers, Number of Heads, Sequence Length, Key Dimension]
Values: [Number of Layers, Number of Heads, Sequence Length, Value Dimension]
Updating the Cache:

At each time step, the new key and value are added to the cache at the position corresponding to the current sequence length.
6. Capturing and Reusing Computation State

The KV cache effectively captures the state of the model’s computations up to the current point in the sequence. This state can be reused when the initial context (the sequence of tokens provided to the model) remains the same across different runs.

Reusing Computation for Identical Contexts:

Scenario: When generating multiple continuations of the same initial text prompt.

Benefit: By reusing the KV cache for the shared initial context, the model saves computational resources by not recomputing the keys and values for those tokens.

Example:

First Run:

Provide an initial prompt: “Once upon a time”.
Compute and cache the keys and values for these tokens.
Subsequent Runs:

Use the same initial prompt.
Load the previously saved KV cache.
Start generating new tokens without recomputing the initial context.
Efficiency Gains:

Reduced Latency: Speeds up inference since the model skips computations for the shared context.

Resource Optimization: Saves computational power and memory bandwidth.

Key Takeaways

Tokens are the basic units of text processing in language models.

Transformer models use self-attention mechanisms to process tokens, relying on queries, keys, and values.

Autoregressive decoding generates text one token at a time, each time conditioned on all previous tokens.

The KV cache stores the keys and values of previous tokens to avoid redundant computations during inference.

Reusing the KV cache for identical initial contexts saves computational resources and speeds up inference.