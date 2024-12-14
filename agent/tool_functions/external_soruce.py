from tavily import TavilyClient
import os
from dotenv import load_dotenv

def fetch_top_sources(query):
    """
    Fetch the top two scored sources for a given query and return them in a JSON structure.
    Args:
        query (str): The query to search for.
    Returns:
        dict: A JSON object containing the top two scored sources and combined content.
    """
    # Load environment variables
    load_dotenv()

    # Initialize TavilyClient with API Key
    api_key = os.getenv("TAVILY_API")
    tavily_client = TavilyClient(api_key=api_key)

    # Perform search query
    response = tavily_client.search(query)

    if response and "results" in response:
        # Sort results by score in descending order
        sorted_results = sorted(response["results"], key=lambda x: x.get("score", 0), reverse=True)
        # Select the top two scored sources
        top_sources = sorted_results[:2]
        
        # Combine content and prepare JSON structure
        combined_content = "\n\n".join(source.get("content", "No Content") for source in top_sources)
        # top_sources_info = [
        #     {"title": source.get("title", "No Title"), "url": source.get("url", "No URL")}
        #     for source in top_sources
        # ]

        # Create JSON response
        result_json = {
            "query": query,
            "combined_content": combined_content,
        }
        return result_json
    else:
        # Handle case when no results are found
        return {
            "query": query,
            "combined_content": "No information found.",
            "top_sources": [],
        }


# Example Usage
# if __name__ == "__main__":
#     query = "What is the current market trend in Real Estate?"
#     result = fetch_top_sources(query)
#     print(result)
