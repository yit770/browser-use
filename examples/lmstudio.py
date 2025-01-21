"""
Example of using Browser Use with LMStudio.
Make sure you have LMStudio running locally with the API server enabled.
"""

import asyncio
from browser_use import Browser, BrowserConfig, Agent
from browser_use.llms import LMStudioLLM
import logging
import os

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    # Initialize LMStudio client
    # You can customize these parameters in your .env file
    llm = LMStudioLLM(
        temperature=0.7,
        max_tokens=512,
        context_length=int(os.getenv("LMSTUDIO_CONTEXT_LENGTH", "8192"))
    )
    
    # Initialize browser with headless=False to see what's happening
    browser = Browser(
        config=BrowserConfig(
            headless=False,  # Set to True in production
            disable_security=True  # Needed for some websites
        )
    )
    
    try:
        # Example 1: Simple browsing
        print("\n=== Example 1: Simple Browsing ===")
        agent = Agent(
            task="Go to google.com and tell me what you see",
            llm=llm,
            browser=browser,
            validate_output=False,  # Disable output validation for now
            max_failures=1  # Reduce max failures to see error faster
        )
        try:
            result = await agent.run()
            logger.debug(f"Agent result type: {type(result)}")
            logger.debug(f"Agent result attributes: {dir(result)}")
            logger.debug(f"Agent result: {result}")
            
            if result and result.history:
                for step in result.history:
                    logger.debug(f"Step: {step}")
                    if hasattr(step, 'output'):
                        print(f"Step output: {step.output}")
                    if hasattr(step, 'error'):
                        print(f"Step error: {step.error}")
                
                final_result = result.final_result()
                if final_result:
                    print(f"Final result: {final_result}")
                    
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}", exc_info=True)
            
    finally:
        # Always close the browser when done
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
