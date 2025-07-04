import logging
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun

# Set up logging for error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Ollama model
llm = Ollama(model="mistral")

# Initialize Wikipedia and ArXiv tools
wikipedia = WikipediaAPIWrapper()
arxiv = ArxivQueryRun()

# Define tools for the agent
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Fetches general knowledge and context from Wikipedia for broad topics."
    ),
    Tool(
        name="ArXiv",
        func=arxiv.run,
        description="Fetches academic papers from ArXiv for technical and research-based queries."
    )
]

# Set up memory to retain conversation context
memory = ConversationBufferMemory(memory_key="chat_history")

# Define prompt template for the agent
prompt_template = PromptTemplate(
    input_pixels=["query", "chat_history"],
    template="""
    You are a Research Agent that answers user queries by combining general knowledge from Wikipedia and academic insights from ArXiv.
    User query: {query}
    Chat history: {chat_history}
    
    Use Wikipedia for broad context and ArXiv for recent academic papers. Summarize findings in concise bullet points in markdown format.
    If no relevant papers are found on ArXiv, note it and rely on Wikipedia or general knowledge.
    """
)

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True
)

# Function to process research query
def process_research_query(query):
    """Process a research query and return a summarized response."""
    try:
        if not query:
            return "Please provide a research query."
        
        # Run the agent with the query
        input_data = {
            "query": query,
            "chat_history": memory.buffer
        }
        response = agent.run(prompt_template.format(**input_data))
        
        # Parse and format the response as markdown bullet points
        try:
            # Attempt to fetch raw data from tools for summarization
            wiki_result = wikipedia.run(query)
            arxiv_result = arxiv.run(query)
            
            # Summarize Wikipedia result (truncate to first 200 characters for brevity)
            wiki_summary = wiki_result[:200] + "..." if wiki_result else "No Wikipedia results found."
            
            # Summarize ArXiv result (list paper titles or note absence)
            arxiv_summary = arxiv_result if arxiv_result else "No relevant papers found on ArXiv."
            
            # Combine results into markdown bullet points
            formatted_response = f"""
## Research Summary for "{query}"

- **Wikipedia Context**: {wiki_summary}
- **ArXiv Papers**: {arxiv_summary}
- **Agent Insights**: {response}
"""
            return formatted_response
        except Exception as e:
            logger.error(f"Error processing tool results: {e}")
            return f"""
## Research Summary for "{query}"

- **Error**: Unable to fetch results due to an issue: {str(e)}
- **Agent Insights**: {response}
"""
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"An error occurred while processing the query: {str(e)}"

# Example usage
if __name__ == "__main__":
    query = input("Enter your research query (e.g., 'recent trends in machine learning'): ")
    result = process_research_query(query)
    print(result)