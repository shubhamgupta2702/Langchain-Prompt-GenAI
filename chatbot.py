from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='meta-llama/Llama-3.2-1B-Instruct',
  task='text-generation',
)

model = ChatHuggingFace(llm=llm)

chat_history = [
  SystemMessage(content="You are a helpful AI Assistant"),
]

while True:
  user = input("You: ")
  chat_history.append(HumanMessage(content=user))
  if user == 'exit':
    break
  result = model.invoke(chat_history)
  chat_history.append(AIMessage(content=result.content))
  print("AI: ", result.content)
  
print(chat_history)