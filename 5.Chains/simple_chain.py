from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

Prompt = PromptTemplate(
    template='Genrate 5 intresting fact about{topic}',
    input_variables=['topic']
)

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = Prompt | model | parser

result = chain.invoke({'topic' : 'cricket'})

print(result)



