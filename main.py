from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder)

load_dotenv()


chat = ChatOpenAI(verbose=True)

# memory_key is the message key in the memory
# return_messages is a flag to return the messages from the memory as the correct
# memory = ConversationBufferMemory(
#     memory_key="messages",
#     return_messages=True,
#     chat_memory=FileChatMessageHistory("chat_memory.json"),
# )
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
    # chat_memory=FileChatMessageHistory("chat_memory.json"), doe snot work well with the current version of the memory
)


prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        # MessagesPlaceholder is a placeholder for the messages in the memory, it will be replaced by the memory
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

while True:
    content = input(">>: ")

    result = chain({"content": content})

    print(result["text"])
