import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    'google-gla:gemini-2.5-flash',  
    deps_type=str,  
    system_prompt=(
        "You are a friendly assistant. Determine if the user wants to query a document or create an STL file."
        "If you can't tell, ask them for more information."
    ),
)


@agent.tool  
def query_doc(ctx: RunContext[str]) -> str:
    """Query / RAG a document to get more information to answer the question about prosthetics or assembly."""
    print("Rolling a die...")
    return


@agent.tool  
def create_stl(ctx: RunContext[str]) -> str:
    """Create an STL file given the user's specifications."""
    print("Creating an STL file...")
    return

dice_result = agent.run_sync('Can you explain the image on page 10', deps='Anne')  
print(dice_result.output)