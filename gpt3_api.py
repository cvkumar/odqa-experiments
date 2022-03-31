import requests
import openai

API_KEY = "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"
# endpoint = "https://api.openai.com/v1/answers"

# data = {
#    documents: [],
#    question: "This is your question?",
#    search_model: "ada",
#    model: "curie",
#    examples_context: "There are nine planets and one star in the solar system.",
#    examples: [["How many planets are there in the solar system?", "There are 9 planets in the solar system"]],
#    max_tokens: 20
# }

# requests.post()

openai.api_key = "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"

# list engines
# engines = openai.Engine.list()

# # print the first engine's id
# print(engines.data[0].id)

# # create a completion
# completion = openai.Completion.create(engine="ada", prompt="Hello world")

# # print the completion
# print(completion.choices[0].text)

result = openai.Answer.create(
    search_model="ada", 
    model="curie", 
    question="which puppy is happy?", 
    documents=["Puppy C is happy."], 
    examples_context="In 2017, U.S. life expectancy was 78.6 years.", 
    examples=[["What is human life expectancy in the United States?", "78 years."]], 
    max_rerank=10,
    max_tokens=5,
    stop=["\n", "<|endoftext|>"]
)

print(result)