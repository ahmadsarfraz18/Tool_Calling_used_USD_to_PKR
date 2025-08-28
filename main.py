from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, function_tool
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. ")

external_client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model= "gemini-2.5-flash",
    openai_client= external_client
)

config= RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled=True
)

@function_tool
def multiplication(a: int, b: int) -> int:
    return a * b

@function_tool
def usd_to_pkr(query: str) -> str:
    return "To convert USD to PKR, multiply the amount in USD by 280. "


agent= Agent(
    name= "CurrencyConverterAgent",
    instructions= "Yor are a helpful assistant that helps the user with their queries and also use the tools if needed. ",
    tools= [multiplication, usd_to_pkr],
)


result= Runner.run_sync(
    agent,
    "Convert 100 USD to PKR . ",
    run_config= config
)

print(result.final_output)
