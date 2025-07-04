# Research Trend Assistant

A Python-based Research Trend Assistant powered by LangChain and the Mistral model (via Ollama). This project enables users to query recent trends and topics by combining general knowledge from Wikipedia and academic insights from ArXiv. The assistant processes natural language inputs, retrieves relevant information, and presents concise summaries in markdown format, leveraging conversational memory for context-aware interactions.

## Features
- Natural Language Queries: Process queries like "recent trends in machine learning" using LangChain and Mistral.
- Dual Knowledge Sources: Combines broad context from Wikipedia and academic papers from ArXiv.
- Conversational Memory: Retains chat history for seamless follow-up questions using LangChain’s ConversationBufferMemory.
- Markdown Output: Summarizes findings in clear, bullet-pointed markdown format.
- Error Handling: Includes robust logging for reliable operation and debugging.

Example Output

### Research Summary for "recent trends in machine learning"

- **Wikipedia Context**: Machine learning is a field of AI focused on algorithms that learn from data...
- **ArXiv Papers**: Recent papers include "Advances in Neural Networks" and "Efficient Training Methods"...
- **Agent Insights**: Key trends involve transformer models, federated learning, and ethical AI considerations.

Requirements

Python 3.8+
Libraries: langchain, langchain_community
Ollama with Mistral model
