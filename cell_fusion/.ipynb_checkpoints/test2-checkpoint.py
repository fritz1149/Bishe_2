from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device=torch.device("cuda")

class QwenChatbot:
    def __init__(self, model_name="Qwen3-4B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # print(self.model.loss_type)
        print(self.model)
        self.model = self.model.to(device)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"User: {text}")

        # tokenizer将思考部分略去
        inputs = self.tokenizer(text, return_tensors="pt")
        # print(f"after tokenization: {self.tokenizer.decode(inputs["input_ids"][0].tolist(), skip_special_tokens=True)}")
        inputs = inputs.to(device)
        # response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        # response = self.tokenizer.decode(response_ids, skip_special_tokens=False)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries? /no_think"
    # print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"chat1: {response_1}")
    print("----------------------")

    # # Second input with /no_think
    # user_input_2 = "Then, how many r's in blueberries? /no_think"
    # # print(f"User: {user_input_2}")
    # response_2 = chatbot.generate_response(user_input_2)
    # print(f"chat2: {response_2}") 
    # print("----------------------")

    # # Third input with /think
    # user_input_3 = "Really? /think"
    # print(f"User: {user_input_3}")
    # response_3 = chatbot.generate_response(user_input_3)
    # print(f"Bot: {response_3}")
