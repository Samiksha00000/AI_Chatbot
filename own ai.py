import ollama
print("Welcome to AI Chatbot")
print("Type exit to stop\n")

while True:
    user_input=input("Yes: ")
    if user_input.lower()=="exit":
        print("Good Bye")
        break
response=ollama.chat(
    model="llama3.2",
    messages=[
        
             {  "role": "user",
                "content": user_input
             }
    ]
)