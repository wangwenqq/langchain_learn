from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.openai_config import openai_setting

chat = ChatOpenAI(openai_api_key=openai_setting['openai_api_key'],
                  openai_api_base=openai_setting['openai_api_base'])

prompt_template = ChatPromptTemplate.from_template("你是一个诗人 请用{style}的风格作诗 内容围绕{content}")
prompt_value = prompt_template.invoke({"style": "豪放", 'content': '月亮'})
print(prompt_value)
print(prompt_value.to_messages())
print(prompt_value.to_string())

message = chat.invoke(prompt_value)
print(message)

chain = prompt_template | chat

output_parser = StrOutputParser()
# Input -(dict)-> PromptTemplate -(promptValue)->ChatModel -(chatMessages)-> StrOutputParser -(string)->Result
print(output_parser.invoke(message))

chain2 = prompt_template | chat | output_parser
# print(chain2.invoke({'style': '浪漫', 'content': '月亮'}))
