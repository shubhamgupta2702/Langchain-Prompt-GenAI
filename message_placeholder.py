from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()

chat_template = ChatPromptTemplate([
  ('system','You are a helpful customer support agent'),
  MessagesPlaceholder(variable_name="chat_history"),
  ('human','{query}')
])
chats_history = []
with open('chat_history.txt') as f:
  chats_history.extend(f.readlines())
  
print(chats_history)

prompt = chat_template.invoke({'chat_history':chats_history, 'query':'Where is my refund?'})

print(prompt)

llm = HuggingFaceEndpoint(
  repo_id='meta-llama/Llama-3.2-1B-Instruct',
  task='text-generation',
)

model = ChatHuggingFace(llm=llm)

result = model.invoke(prompt)

print(result.content)