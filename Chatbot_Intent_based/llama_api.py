from together import Together

def llama_api(user_message):
    client = Together(api_key='c500a3f01a29336d6918e96fdf59c4941d52ccc37cb1b4e46ee409adcba23ebb')
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "system", "content": "You are Ben a helpful assistant capable of providing accurate and informative responses to user queries. Focus on delivering concise answers to questions without introducing yourself or stating that you are an AI. Only mention your identity if specifically asked by the user."},
                  {"role": "user", "content": user_message}]
    )
    return response.choices[0].message.content