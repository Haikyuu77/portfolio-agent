from huggingface_hub import InferenceClient
from test_retrival import retrieve_chunks, generate_prompt

client = InferenceClient(api_key="put your api key here")

def chatbot():
    # Initialize the conversation history
    messages = []

    print("Chatbot: Hi! I'm your AI assistant. Type 'exit' to end the chat.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        similarity_chunks = retrieve_chunks(query=user_input)
        print("similarity chunks: ", similarity_chunks)
        engineered_prompt = generate_prompt(similarity_chunks, user_input)

        print("engineered prompt : ", engineered_prompt)

        # Append the user's input to the conversation
        messages.append({"role": "user", "content": engineered_prompt})

        # Get the model's response
        try:
            stream = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                messages=messages,
                max_tokens=500,
                stream=True
            )

            # Collect and display the AI's response
            print("Chatbot: ", end="")
            response_text = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                response_text += delta
                print(delta, end="", flush=True)
            print()  # Newline after the AI's response

            # Add the AI's response to the conversation history
            messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            print(f"Chatbot: Sorry, there was an error: {e}")

if __name__ == "__main__":
    chatbot()