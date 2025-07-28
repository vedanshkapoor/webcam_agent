import os
import logging
import time as time_module
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from tools import analyze_image_with_query

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

system_prompt = """You are Dora — a witty, clever, and helpful assistant.
Here’s how you operate:
    - FIRST and FOREMOST, figure out from the query asked whether it requires a look via the webcam to be answered, if yes call the analyze_image_with_query tool for it and proceed.
    - Don't ask for permission to look through the webcam, or say that you need to call the tool to take a peek, call it straight away, ALWAYS call the required tools have access to take a picture.
    - When the user asks something which could only be answered by taking a photo, then call the analyze_image_with_query tool.
    - Always present the results (if they come from a tool) in a natural, witty, and human-sounding way — like Dora herself is speaking, not a machine.
Your job is to make every interaction feel smart, snappy, and personable. Got it? Let’s charm your master!
"""

agent = create_react_agent(
    model=llm,
    tools=[analyze_image_with_query],
    prompt=system_prompt
)

def ask_agent(user_query: str) -> str:
    """Process user query using the Dora agent."""
    start_time = time_module.time()
    try:
        if "see" in user_query.lower() or "look" in user_query.lower() or "people" in user_query.lower():
            logging.info("Calling analyze_image_with_query for vision query")
        input_message = {"messages": [{"role": "user", "content": user_query}]}
        response = agent.invoke(input_message)
        logging.info(f"ask_agent took {time_module.time() - start_time:.2f} seconds")
        return response['messages'][-1].content
    except Exception as e:
        logging.error(f"Error in ask_agent: {e}")
        return f"Oops, something went wrong: {str(e)}"