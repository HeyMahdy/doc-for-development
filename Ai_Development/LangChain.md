

---

# 🚀 LangChain Model Wrappers Cheat Sheet

| **Wrapper Class**          | **Provider / Platform**                                                                            | **Example Models**                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **ChatOpenAI**             | OpenAI + any **OpenAI-compatible APIs** (HuggingFace router, Fireworks, DeepInfra, Together, etc.) | `"gpt-4o"`, `"gpt-3.5-turbo"`, `"meta-llama/Meta-Llama-3-8B-Instruct"`           |
| **ChatAnthropic**          | [Anthropic](https://www.anthropic.com/)                                                            | `"claude-3-opus"`, `"claude-3-sonnet"`, `"claude-3-haiku"`                       |
| **ChatMistralAI**          | [Mistral](https://mistral.ai/) API                                                                 | `"mistral-small"`, `"mistral-medium"`, `"mistral-large"`                         |
| **ChatCohere**             | [Cohere](https://cohere.com/)                                                                      | `"command-r"`, `"command-r-plus"`                                                |
| **ChatGoogleGenerativeAI** | Google Gemini (PaLM successor)                                                                     | `"gemini-1.5-pro"`, `"gemini-1.5-flash"`                                         |
| **ChatPerplexity**         | [Perplexity AI](https://www.perplexity.ai/)                                                        | `"pplx-7b-chat"`, `"pplx-70b-chat"`                                              |
| **ChatTogether**           | [Together.ai](https://www.together.ai/)                                                            | `"togethercomputer/llama-2-70b-chat"`, `"NousResearch/Nous-Hermes-2-Mistral-7B"` |
| **ChatOllama**             | [Ollama](https://ollama.ai/) (local models)                                                        | `"llama3"`, `"mistral"`, `"codellama"`                                           |
| **ChatLiteLLM**            | [LiteLLM](https://github.com/BerriAI/litellm) universal proxy                                      | Can proxy to OpenAI, Anthropic, Mistral, HuggingFace, etc.                       |

---

### 🔹 Non-Chat LLMs (simple completions)

| **Wrapper Class**  | **Provider**                        | **Example Models**                              |
| ------------------ | ----------------------------------- | ----------------------------------------------- |
| **OpenAI**         | OpenAI (or OpenAI-compatible)       | `"text-davinci-003"`, `"gpt-4o-mini"`           |
| **Cohere**         | Cohere                              | `"command-xlarge-nightly"`                      |
| **AI21**           | AI21 Labs                           | `"j2-grande-instruct"`, `"j2-jumbo-instruct"`   |
| **HuggingFaceHub** | Hugging Face Inference API          | `"google/flan-t5-xl"`, `"tiiuae/falcon-40b"`    |
| **Ollama**         | Ollama local completions            | `"llama3"`, `"mistral"`                         |
| **Replicate**      | [Replicate](https://replicate.com/) | `"meta/llama-2-70b"`, `"stability-ai/stablelm"` |

---

### 🔹 Embeddings (turn text → vectors)

| **Wrapper Class**                | **Provider**               | **Example Models**                                     |
| -------------------------------- | -------------------------- | ------------------------------------------------------ |
| **OpenAIEmbeddings**             | OpenAI                     | `"text-embedding-3-large"`, `"text-embedding-3-small"` |
| **CohereEmbeddings**             | Cohere                     | `"embed-english-v3.0"`                                 |
| **HuggingFaceEmbeddings**        | Local HF models            | `"sentence-transformers/all-mpnet-base-v2"`            |
| **HuggingFaceHubEmbeddings**     | Hugging Face Inference API | `"BAAI/bge-large-en-v1.5"`                             |
| **GoogleGenerativeAIEmbeddings** | Google Gemini              | `"embedding-001"`                                      |
| **MistralAIEmbeddings**          | Mistral                    | `"mistral-embed"`                                      |
| **OllamaEmbeddings**             | Ollama                     | `"nomic-embed-text"`                                   |
| **VertexAIEmbeddings**           | Google Vertex AI           | `"textembedding-gecko@001"`                            |

---

⚡ **Rule of thumb**:

* If you see **`ChatXXX`** → use it when the provider supports chat-style APIs.
* If just **`XXX`** (no "Chat") → it’s for plain text completions.
* If **`XXXEmbeddings`** → it’s for embeddings.

---

