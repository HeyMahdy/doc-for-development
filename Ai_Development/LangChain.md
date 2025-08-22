

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



# Popular Built-in Tools in LangChain

**Estimated time:** 5 minutes  

LangChain offers a variety of built-in tools designed to enhance AI agent capabilities across search, coding, web browsing, productivity, and more. These tools work seamlessly with language models, allowing for dynamic task execution.  

This guide explores popular tools and toolkits categorized by use case, including notes on availability and pricing.

---

## Tools and Toolkits

- **Tool** → A utility designed to be called by a model (input/output optimized for LLMs).  
- **Toolkit** → A collection of tools meant to be used together.  

⚠️ Some tools are free, others require payment. Always check official documentation.

---

## 🔎 Search Tools

| Tool/Toolkit   | Function        | Purpose |
|----------------|-----------------|---------|
| **SerpAPI**    | Web search      | Performs web searches and returns answers |
| **Google Search** | Web search  | Executes Google searches and returns URLs, snippets, and titles |
| **Tavily Search** | AI-optimized search | Returns URLs, content, titles, images, and answers |
| **Wikipedia**  | Knowledge base search | Searches Wikipedia articles and returns summaries |

---

## 📊 Code Interpretation & Data Analysis

| Tool/Toolkit             | Function            | Purpose |
|--------------------------|---------------------|---------|
| **Python REPL**          | Code execution      | Executes Python code for calculations, data analysis, automation |
| **Pandas DataFrame**     | Data manipulation   | Interact with and analyze tabular data |
| **SQL Database Toolkit** | Database querying   | Query and manipulate SQL databases using natural language |
| **LLMMathChain**         | Mathematical computation | Solves math problems by translating to Python |
| **JSON Toolkit**         | JSON manipulation   | Interact with large JSON/dictionary objects efficiently |

---

## 🌐 Web Browsing & Interaction

| Tool/Toolkit       | Function         | Purpose |
|--------------------|------------------|---------|
| **Requests Toolkit** | HTTP requests   | Interact with web APIs and fetch web content |
| **PlayWright Browser** | Browser automation | Navigate and interact with websites |
| **MultiOn Toolkit**   | Web app interaction | Interact with popular web applications |
| **ArXiv**             | Paper search   | Retrieve scientific papers from ArXiv |

---

## 📅 Productivity & Collaboration

| Tool/Toolkit        | Function             | Purpose |
|---------------------|----------------------|---------|
| **Gmail Toolkit**   | Email management     | Read, send, manage Gmail emails |
| **Office365 Toolkit** | Office suite integration | Work with Outlook, OneDrive, etc. |
| **Slack Toolkit**   | Team communication   | Send/read Slack messages |
| **Github Toolkit**  | Repo management      | Manage GitHub repos, issues, PRs |
| **Google Calendar** | Calendar management  | Create, read, update events |

---

## 📂 File & Document Processing

| Tool/Toolkit      | Function          | Purpose |
|-------------------|------------------|---------|
| **File System**   | Local file ops    | Read, write, manage files |
| **Google Drive**  | Cloud storage     | Access and manage Drive files |
| **VectorStoreQA** | Document querying | Query from vector databases |
| **Document Loaders** | Content extraction | Extract from PDF, DOCX, etc. |

---

## 💰 Financial & Business Tools

| Tool/Toolkit      | Function             | Purpose |
|-------------------|----------------------|---------|
| **Yahoo Finance** | Financial news       | Retrieve news & market info |
| **GOAT**          | Transactions         | Payments, purchases, investments |
| **Polygon IO**    | Market data          | Real-time & historical stock/option data |
| **Stripe**        | Payment processing   | Manage payments, subscriptions |

---

## 🤖 AI & Machine Learning Integration

| Tool/Toolkit          | Function          | Purpose |
|-----------------------|------------------|---------|
| **Dall-E**            | Image creation   | Generate images from text |
| **HuggingFace Hub**   | Model access     | Access ML models on HuggingFace |
| **Google Imagen**     | Image generation | Vertex AI image generation |
| **Nuclia Understanding** | Data indexing | Index unstructured data for retrieval |

---

## 📌 Summary

- LangChain integrates **tools** and **toolkits** to extend LLMs across search, data analysis, web browsing, productivity, finance, and AI integration.  
- Tools serve specific functions (e.g., **SerpAPI** = search, **Python REPL** = code execution).  
- Toolkits group related tools (e.g., **SQL Toolkit**, **Gmail Toolkit**).  
- Some tools are **free**, others **paid** — always confirm via official docs.  
- Specialized tools (e.g., **Tavily** for AI search, **MultiOn** for app interaction) make LangChain flexible for business and research use cases.


