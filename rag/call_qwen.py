from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "not used"
openai_api_base = ""
import requests


def get_response(input_text):  
    

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen3-32B",
        messages=[
            {"role": "user", "content": input_text},
        ],
        max_tokens=4096,
        temperature=1,
        top_p=1,
        presence_penalty=1.0,
        extra_body={
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return chat_response.choices[0].message.content

def get_response_from_gpt4(input_text):
    gpt4_key = ""

    client = OpenAI(api_key=gpt4_key)
    Model = "gpt-4o"
    response = client.chat.completions.create(
        model=Model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ],
        temperature=0,
    )

    return response.choices[0].message.content

def get_response_from_ds(input_text):
    api_token_ds = ""
    client = OpenAI(api_key=api_token_ds, base_url="https://api.deepseek.com/v1")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": input_text},
        ],
        stream=False
    )

    return response.choices[0].message.content

def get_response_from_61(input_text):
    response = requests.post(
        "",
        json={"message": input_text}
    )

    return response.json()['response']

if __name__ == "__main__":
    query = "hello world？"
    answer = get_response(query)
    print(answer)