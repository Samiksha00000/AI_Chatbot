import ollama

print("Welcome to Chatbot!")

print("Type 'exit' to stop the chatbot.\n")

messages = []

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":

        break

    messages.append({"role": "user", "content": user_input})

    response = ollama.chat(

        model="llama3.2",

        messages=messages

    )

    print("ChatBot:", response["message"]["content"])

    messages.append({
        "role": "assistant",
        "content": response["message"]["content"]
    })