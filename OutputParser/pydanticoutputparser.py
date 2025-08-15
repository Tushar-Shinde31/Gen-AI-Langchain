from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class ElectionLeader(BaseModel):
    name: str = Field(description='Name of the election leader')
    age: int  = Field(gt=18, description='Age of the election leader')
    city: str = Field(description='City in Maharashtra the leader is from')

parser = PydanticOutputParser(pydantic_object=ElectionLeader)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional class election leader in Maharashtra, specifically from {city}.\n{format_instructions}',
    input_variables=['city'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

city = input("Which city in Maharashtra do you want? ")

chain = template | model | parser

final_result = chain.invoke({'city': city})

print(final_result)

