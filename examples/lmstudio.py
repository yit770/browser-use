"""
Example of using Browser Use with LMStudio.
Make sure you have LMStudio running locally with the API server enabled.
"""

import asyncio
import time
from browser_use import Agent, Browser, BrowserConfig
from browser_use.llms import LMStudioLLM
import logging
import os

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def sleep_with_message(seconds: int, message: str):
    print(f"\n{'='*50}")
    print(f"{message}")
    print(f"Waiting {seconds} seconds...")
    print("=" * 50)
    await asyncio.sleep(seconds)


async def main():
    print("\n=== Starting Process ===")
    await sleep_with_message(1, "Initializing process...")

    # Initialize LMStudio client
    print("\n=== LMStudio Initialization ===")
    browser_config = BrowserConfig(
        headless=False, disable_security=True  # הפעלת הדפדפן במצב גלוי
    )

    browser = Browser(config=browser_config)

    llm = LMStudioLLM(
        temperature=0.7,
        max_tokens=512,
        context_length=int(os.getenv("LMSTUDIO_CONTEXT_LENGTH", "8192")),
        browser=browser,
    )
    await sleep_with_message(1, "LMStudio initialized successfully")

    try:
        print("\n=== Creating Agent ===")
        agent = Agent(
            task="Go to example.com and tell me what you see",
            llm=llm,
            max_failures=1,
        )
        await sleep_with_message(0, "Agent created successfully")

        try:
            print("\n=== Starting Task Execution ===")
            result = await agent.run(max_steps=3)

            if result and result.history:
                print("\n=== Execution Steps ===")
                for i, step in enumerate(result.history, 1):
                    print(f"\nStep {i}:")
                    print("-" * 30)
                    if hasattr(step, "output"):
                        print(f"Output: {step.output}")
                    if hasattr(step, "error"):
                        print(f"Error: {step.error}")
                    await asyncio.sleep(0)  # השהייה כדי לראות את התהליך

                final_result = result.final_result()
                if final_result:
                    print("\n=== Final Result ===")
                    print(f"{final_result}")

        except Exception as e:
            print("\n=== Error! ===")
            logger.error(f"Error running Agent: {str(e)}", exc_info=True)

    finally:
        print("\n=== Process Complete ===")
        await sleep_with_message(1, "Process finished")

        # משאיר את הדפדפן פתוח כדי שתוכל לראות את התהליך
        print("\n=== Browser is still open, press Enter to close ===")
        input()
        await browser.close()  # סגירה ידנית לאחר צפייה


if __name__ == "__main__":
    asyncio.run(main())
