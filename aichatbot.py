import ollama

print("🤖 AI Chatbot")
print("Type 'exit' to stop\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ]
    )

    print("AI:", response["message"]["content"])