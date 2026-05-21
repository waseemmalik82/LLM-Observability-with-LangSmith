from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['langchain_project'] = 'sequential app'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
model2 = ChatOpenAI(model='gpt-4o', temperature=0.7)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model2 | parser

config = {
    'run_name': 'sequential chain',
    'tags': ['llm app', 'report_genration', 'summrization'],
    'metadata': {'model':'gpt-4o-mini'}
}

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)
