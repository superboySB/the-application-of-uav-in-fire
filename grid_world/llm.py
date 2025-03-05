import openai
import tiktoken
import time
import hashlib

# 设置 API 密钥和 API 基础 URL
openai.api_key = "sk-tVIu0cYO13rAFxiB4aF1C0Be68464c0fB4E75f25EfA357Fb"
openai.api_base = "https://api.lqqq.ltd/v1"

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

# 简单的缓存字典
cache = {}

def hash_messages(messages):
    """生成消息的哈希值，用于缓存键"""
    message_str = ''.join([f"{msg['role']}: {msg['content']}" for msg in messages])
    return hashlib.md5(message_str.encode('utf-8')).hexdigest()

def GPT_response(messages, model_name):
    token_num_count = 0

    # 计算消息的哈希值
    cache_key = hash_messages(messages)

    # 检查缓存
    if cache_key in cache:
        print("Cache hit!")
        return cache[cache_key]

    for item in messages:
        token_num_count += len(enc.encode(item["content"]))

    if model_name in ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613']:
        try:
            result = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=30
            )
        except Exception as e:
            print("Exception", e)
            try:
                print(f'{model_name} Waiting 60 seconds for API query')
                time.sleep(60)
                result = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            except:
                print(token_num_count)
                return 'Out of tokens', token_num_count
        token_num_count += len(enc.encode(result.choices[0]['message']['content']))
        print(f'Token_num_count: {token_num_count}')

        # 将结果存入缓存
        cache[cache_key] = (result.choices[0]['message']['content'], token_num_count)
        return result.choices[0]['message']['content'], token_num_count

    else:
        raise ValueError(f'Invalid model name: {model_name}')


