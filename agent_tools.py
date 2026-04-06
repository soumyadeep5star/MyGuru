from langchain_core.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


def _format_vector_results(
    heading: str,
    results,
    empty_message: str,
    max_results: int = 3,
) -> str:
    """Format top vector-search results without hard score cutoffs."""
    if not results:
        return empty_message

    output = f"{heading}:\n\n"
    for i, result in enumerate(results[:max_results], 1):
        logger.info(
            "Vector result %s: %s (type: %s, distance: %.4f, relevance: %.2f%%)",
            i,
            result.get('title'),
            result.get('type'),
            result.get('distance', 0),
            result.get('relevance_score', 0) * 100,
        )
        output += f"{i}. **{result.get('title')}**\n"
        output += f"   {result.get('text', '')[:700]}\n\n"
        if result.get('type'):
            output += f"   Type: {result.get('type')}\n"
        if result.get('url'):
            output += f"   Source: {result.get('url')}\n"
        output += "\n"

    return output


class ConfluenceSearchInput(BaseModel):
    """Input for Confluence search tool"""
    query: str = Field(description="The search query to find relevant Confluence documentation")


class ConfluenceSearchTool(BaseTool):
    """Tool for searching Confluence documentation"""
    name: str = "confluence_search"
    description: str = """Useful for searching internal Confluence documentation, wiki pages, and PDF attachments.
    Use this tool when the user asks about:
    - Internal documentation
    - Company processes
    - Technical specifications
    - Project documentation
    - Knowledge base articles
    Input should be a search query string."""
    args_schema: type[BaseModel] = ConfluenceSearchInput
    vector_store: object = None
    
    def _run(self, query: str) -> str:
        """Search Confluence documentation"""
        try:
            logger.info(f"Confluence Tool searching for: '{query}'")
            if not self.vector_store:
                return "Confluence search is not available. Vector store not initialized."
            
            # Search using vector store
            results = self.vector_store.search(query, k=5)
            logger.info(f"Confluence Tool found {len(results)} results")
            
            if not results:
                logger.info("Confluence Tool: No results found")
                return "No relevant information found in Confluence documentation."
            
            output = _format_vector_results(
                heading="Confluence Documentation",
                results=results,
                empty_message="No relevant information found in Confluence documentation.",
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Confluence search: {e}")
            return f"Error searching Confluence: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class GitHubSearchInput(BaseModel):
    """Input for GitHub search tool"""
    query: str = Field(description="The search query to find relevant GitHub repositories")


class GitHubVectorSearchTool(BaseTool):
    """Tool for searching GitHub content from vector database"""
    name: str = "github_search"
    description: str = """Useful for searching indexed GitHub repositories, README files, and code contents.
    Use this tool when the user asks about:
    - Code examples
    - Open source projects
    - Repository information
    - Implementation details in code
    - Indexed GitHub content stored in the vector database
    Input should be a search query string."""
    args_schema: type[BaseModel] = GitHubSearchInput
    vector_store: object = None
    
    def _run(self, query: str) -> str:
        """Search GitHub content from vector database"""
        try:
            logger.info(f"GitHub Tool searching for: '{query}'")
            if not self.vector_store:
                return "GitHub search is not available. GitHub vector store not initialized."

            results = self.vector_store.search(query, k=3)
            logger.info(f"GitHub Tool found {len(results)} vector results")

            output = _format_vector_results(
                heading="GitHub Repository Content",
                results=results,
                empty_message="No relevant GitHub content found in the indexed vector database.",
            )

            return output
            
        except Exception as e:
            logger.error(f"Error in GitHub search: {e}")
            return f"Error searching GitHub: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class DatabaseSearchInput(BaseModel):
    """Input for database search tool"""
    sql_query: str = Field(description="SQL SELECT query to execute on the Azure SQL database")


class DatabaseSearchTool(BaseTool):
    """Tool for searching Azure SQL Database"""
    name: str = "database_search"
    description: str = """Useful for querying structured data from Azure SQL Database.
    Use this tool when the user asks about:
    - Structured data, records, or database information
    - Specific data queries that need SQL
    - Information not found in documentation or code
    
    Input should be a valid SQL SELECT query. The tool will execute it and return results.
    The database schema will be provided to help you construct queries."""
    args_schema: type[BaseModel] = DatabaseSearchInput
    database_searcher: object = None
    schema_info: str = ""
    
    def _run(self, sql_query: str) -> str:
        """Execute SQL query on database"""
        try:
            logger.info(f"Database Tool executing query: {sql_query[:100]}...")
            if not self.database_searcher:
                return "Database search is not available. Please configure Azure SQL Database credentials."
            
            # Execute query
            result = self.database_searcher.search(sql_query)
            logger.info(f"Database query completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in database search: {e}")
            return f"Error querying database: {str(e)}"
    
    async def _arun(self, sql_query: str) -> str:
        """Async version"""
        return self._run(sql_query)


def create_confluence_tool(vector_store) -> ConfluenceSearchTool:
    """Create Confluence search tool with vector store"""
    return ConfluenceSearchTool(vector_store=vector_store)


def create_github_tool(vector_store) -> GitHubVectorSearchTool:
    """Create GitHub search tool with vector store"""
    return GitHubVectorSearchTool(vector_store=vector_store)


def create_database_tool(database_searcher):
    """Create database search tool with searcher"""
    try:
        schema_info = database_searcher.get_schema_info() if database_searcher else ""
        return DatabaseSearchTool(
            database_searcher=database_searcher,
            schema_info=schema_info
        )
    except Exception as e:
        logger.error(f"Error creating database tool: {e}")
        return None

