"""
Agent.py
---

A PDF Assistant Agent is created in this file, which uses pgvector
"""

from phi.agent import Agent
from phi.storage.agent.postgres import PgAgentStorage
from phi.model.groq import Groq
from phi.embedder.huggingface import HuggingfaceCustomEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
import os

# LLM Setup
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"  # Change to Your API KEY!
os.environ["HF_TOKEN"] = "HF_TOKEN"  # Change to Your Huggingface Token
groq = Groq(id="llama3-groq-70b-8192-tool-use-preview")

# It will be available as standalone pgVector container
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"



# Creating the Agent
agent = Agent(
    model=groq,
    storage=PgAgentStorage(table_name="recipe_agent", db_url=db_url),
    knowledge_base=PDFUrlKnowledgeBase(
        urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=PgVector2(
            collection="recipe_documents", db_url=db_url, embedder=HuggingfaceCustomEmbedder()
        )
    ),  # Show tool calls in the response
    show_tool_calls=True,  # Enable the agent to search the knowledge base
    search_knowledge=True,  # Enable the agent to read the chat history
    read_chat_history=True,
)

# Comment out after first run
agent.knowledge_base.load(recreate=False)  # type: ignore

agent.print_response("How do I make pad thai?", markdown=True)
