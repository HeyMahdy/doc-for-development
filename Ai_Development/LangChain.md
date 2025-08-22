

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



Popular Built-in Tools in LangChain

Estimated time: 5 minutes

LangChain offers a variety of built-in tools designed to enhance AI agent capabilities across search, coding, web browsing, productivity, and more. These tools work seamlessly with language models, allowing for dynamic task execution. This guide explores popular tools and toolkits categorized by use case, including notes on availability and pricing.

In this reading, you will identify key built-in tools and toolkits available in LangChain and understand the purpose and function of each tool within its use case category. You will also compare multiple search and data analysis tools to determine their ideal applications and evaluate whether a tool requires payment or is freely accessible based on official documentation.
Tools and toolkit

Tools are utilities designed to be called by a model: their inputs are designed to be generated by models, and their outputs are designed to be passed back to models.

A toolkit is a collection of tools meant to be used together.

One thing to keep in mind is that some tools may be paid and some free. It is advised that check if the tool is free or not on the official documentation.
List of built-in tools in Langchain

Below is the list of popular built-in tools in Langchain based on use cases:

    Search tools

Tool/Toolkit 	Function 	Purpose
SerpAPI 	Web search 	Performs web searches and returns answers
Google Search 	Web search 	Executes Google searches and returns URLs, snippets, and titles
Tavily Search 	AI-optimized search 	Search engine built specifically for AI agents. Returns URLs, content, titles, images, and answers
Wikipedia 	Knowledge base search 	Searches Wikipedia articles and returns relevant information and summaries

    Code interpretation and data analysis

Tool/Toolkit 	Function 	Purpose
Python REPL 	Code execution 	Executes Python code for complex calculations, data analysis, and automation
Pandas DataFrame 	Data manipulation 	Enables agents to interact with and analyze tabular data in Pandas DataFrames
SQL Database Toolkit 	Database querying 	Allows agents to query and manipulate SQL databases using natural language. Returns URLs, content, titles, images, and answers
LLMMathChain 	Mathematical computation 	Solves mathematical problems by translating them to Python code and evaluating them
JSON Toolkit 	JSON manipulation 	Helps agents interact with large JSON/dictionary objects efficiently

    Web browsing and interaction

Tool/Toolkit 	Function 	Purpose
Requests Toolkit 	HTTP requests 	Constructs HTTP requests to interact with web APIs and fetch web content
PlayWright Browser 	Browser automation 	Controls web browsers to navigate websites and interact with web pages
MultiOn Toolkit 	Web app interaction 	Enables AI agents to interact with popular web applications
ArXiv 	Scientific paper search 	Searches and retrieves scientific papers from the arXiv repository

    Productivity and collaboration

Tool/Toolkit 	Function 	Purpose
Gmail Toolkit 	Email management 	Allows reading, sending, and managing emails through Gmail
Office365 Toolkit 	Office suite integration 	Interacts with Microsoft 365 applications, including Outlook, OneDrive, etc.
Slack Toolkit 	Team communication 	Enables sending and reading messages in Slack channels and direct messages
Github Toolkit 	Code repository management 	Manages repositories, issues, pull requests, and other GitHub features
Google Calendar 	Calendar management 	Creates, reads, and updates calendar events in Google Calendar

    File and document processing

Tool/Toolkit 	Function 	Purpose
File System 	Local file operations 	Interacts with local file system to read, write, and manage files
Google Drive 	Cloud storage 	Connects to Google Drive to access, search, and manage cloud files
VectorStoreQA 	Document querying 	Queries information from documents stored in vector databases
Document Loaders 	Content extraction 	Extracts content from various document formats (PDF, DOCX, etc.)

    Financial and business tools

Tool/Toolkit 	Function 	Purpose
Yahoo Finance 	Financial news 	Retrieves financial news articles and market information
GOAT 	Financial transactions 	Creates/receives payments, purchases goods, and makes investments
Polygon IO 	Market data 	Provides real-time and historical market data for stocks, options, etc.
Stripe 	Payment processing 	Manages payments, subscriptions, and other e-commerce functions

    AI and machine learning integration

Tool/Toolkit 	Function 	Purpose
Dall-E Image Generator 	Image creation 	Generates images from text descriptions using OpenAI's Dall-E models
HuggingFace Hub Tools 	Model access 	Connects to various machine learning models hosted on HuggingFace
Google Imagen 	Image generation 	Accesses Google's image generation capabilities through Vertex AI
Nuclia Understanding 	Unstructured data indexing 	Indexes unstructured data from various sources for enhanced retrieval
Summary

In this reading, you learned that:

    LangChain integrates tools and toolkits to extend the functionality of language models across diverse use cases such as search, data analysis, web browsing, and productivity.

    Each tool serves a specific function—for example, SerpAPI performs web searches, while the Python REPL executes code for data analysis or automation.

    Toolkits group related tools (for example, SQL Database Toolkit or Gmail Toolkit), enabling more complex task orchestration within a single interface.

    Some tools are free, and others require payment. Always verify the pricing and availability through the official documentation before integration.

    Use case-specific tools (for example, Tavily for AI-optimized search or MultiOn for web app interaction) help tailor LangChain applications to real-world business or research needs.

