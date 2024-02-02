from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder)

load_dotenv()


chat = ChatOpenAI()

# memory_key is the message key in the memory
# return_messages is a flag to return the messages from the memory as the correct
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)


prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
)

while True:
    content = input(">>: ")

    result = chain({"content": content})

    print(result)
