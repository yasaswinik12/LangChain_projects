import warnings
import langchain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# load_dotenv would load the .env file which contains the secret API key of OPENAI
from dotenv import load_dotenv

# Using argparse, we can give input variables as arguments in the command line
import argparse

load_dotenv()

# We are creating two arguments : task and language with their default values set to return a list of numbers and python
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args=parser.parse_args()

# Ignores all the warnings in the command line
warnings.filterwarnings("ignore")

# Initialising the llm model
llm = OpenAI()

# creating prompt for the chaining
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)
# creating prompt for the chaining
check_prompt = PromptTemplate(
    template="Write a test for the following {language} code: \n{code} ",
    input_variables=["language", "code"]
)
# Creating the chain which contains prompt template and the model
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)
check_chain=LLMChain(
    llm=llm,
    prompt=check_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, check_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)
result = chain({
    "language":args.language,
    "task":args.task
})
# result is a dictionary with keys as the input parameters and values as their default or user given values and text as the key whose value is the answer returned by the model
# result={input_variables:default values / user given values
#         'text': 'result'}

print(">>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>> GENERATED TEST:")
print(result["test"])