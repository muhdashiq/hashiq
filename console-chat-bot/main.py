from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

# Loading the environment variable for the API Keys
load_dotenv()

chat = ChatOpenAI(verbose=True)
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("message-history.json"),
#     memory_key="messages",
#     return_messages=True
# )

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
    verbose=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

print("[AI Agent] : Ask me anything? ")
while True:
    content = input("[      Me] : ")
    result = chain({"content": content})
    print("[AI Agent] : " + result["text"])