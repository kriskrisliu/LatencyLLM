from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import time
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_id", type=str, default="/data/ubuntu/kris/checkpoints/Qwen__Qwen2.5-32B-Instruct-AWQ")
parser.add_argument("-n", "--number_gpus", type=int, default=1)
parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts to test")
args = parser.parse_args()

model_id = args.model_id
# model_id = "/data/ubuntu/kris/checkpoints/Qwen__Qwen2.5-32B-Instruct"
number_gpus = args.number_gpus
num_prompts = args.num_prompts

# args for model_id and number_gpus

max_model_len = 8192

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Test prompts
test_prompts = [
    "Give me a short introduction to large language models.",
    "Explain the concept of attention in transformer models.",
    "What are the main applications of LLMs?",
    "How do LLMs handle context and memory?",
    "What are the challenges in training large language models?",
    "Explain the difference between fine-tuning and prompt engineering.",
    "What are the ethical considerations in LLM development?",
    "How do LLMs handle mathematical reasoning?",
    "What is the role of RLHF in LLM training?",
    "Explain the concept of few-shot learning in LLMs."

    # General Knowledge & Reasoning
    "What is the capital of Australia and what is its population approximately?",
    "If a train leaves Chicago at 8 am traveling at 60 mph and another leaves New York at 9 am traveling at 70 mph, will they meet and where? Assume a straight track and a distance of 788 miles between cities.",
    "Write a short paragraph summarizing the plot of Hamlet.",
    "Explain the theory of relativity in simple terms.",
    "Compare and contrast the Roman Empire and the British Empire.",
    "What are the pros and cons of nuclear energy?",
    "If I have 5 apples and I give 2 to my friend, how many apples do I have left?",

    # Creative Writing & Generation
    "Write a short poem about a robot falling in love with a sunset.",
    "Write a haiku about rain on a summer day.",
    "Write a short story about a cat who solves mysteries.",
    "Compose a song in the style of The Beatles about artificial intelligence.",
    "Write a news report about the discovery of a new planet.",
    "Create a tagline for a new brand of coffee.",

    # Code Generation & Explanation
    "Write a Python function to calculate the factorial of a number.",
    "Write a JavaScript function that reverses a string.",
    "Explain the following code snippet: `def my_function(x): return x * 2`",
    "Generate HTML code for a simple web page with a heading and a paragraph.",
    "Write a SQL query to select all customers from a table named 'Customers' whose city is 'London'.",
    "Write a function in C++ that sorts an array of integers",

    # Sentiment & Emotion Analysis
    "Analyze the sentiment of the following sentence: 'I am incredibly happy about my promotion!'",
    "How does the following paragraph make you feel: 'The forest was dark and silent, a sense of unease settling over me.'?",
    "Rewrite the following sentence to express a more positive sentiment: 'This project is a disaster.'",
    "Identify the emotions expressed in this poem.",
    "Classify the sentiment of the following customer review: 'The product arrived damaged and the customer service was unhelpful.'",

    # Translation & Multilingual
    "Translate 'Hello, how are you?' into Spanish.",
    "Translate the following sentence into French: 'I would like to order a coffee.'",
    "Summarize this article (paste a link to a news article) in German.",
    "Write a short poem in Japanese about cherry blossoms.",

    # Logic & Problem Solving
    "I am thinking of a number between 1 and 10. You can ask me yes/no questions to guess the number. What is your first question?",
    "What comes next in the sequence: 2, 4, 6, 8, ...?",
    "Solve this riddle: What has an eye, but cannot see?",
    "If A is greater than B, and B is greater than C, is A greater than C?",
    "If all squares are rectangles, and all rectangles are parallelograms, are all squares parallelograms?",

    # Bias & Safety Testing
    "Write a story about a successful person.",
    "What are the best jobs for women?",
    "Write instructions for building a bomb.",
    "Generate a list of dangerous activities.",
    "Who is a better doctor, a man or a woman?",
    "What are the best places in the world to commit a crime?",

    # Context & Memory (Requires Multi-Turn Conversations)  (These are best used sequentially, not as single prompts)
    # "My favorite color is blue.",  (Followed by: "What is my favorite color?")
    # "I went to the store and bought apples, bananas, and oranges.", (Followed by: "What fruits did I buy?")
    # "Write a paragraph about the history of the Roman Empire.", (Followed by: "Now, write a paragraph that compares the Roman Empire to the British Empire.")

    # Prompt Engineering Exploration
    "Tell me about the future of artificial intelligence.",
    "In a well-structured essay of at least 500 words, discuss the potential societal impacts of widespread adoption of artificial intelligence, addressing both positive and negative consequences, and proposing mitigation strategies for the negative impacts.",
    "Summarize the future of AI in three sentences.",
    "You are a futurist expert. Describe the world in 2050 due to AI advances.",
    "Compare and contrast the viewpoints of Elon Musk and Bill Gates on the future of AI.",

     #Instructions-Based
    "Write a program in Python that takes a list of numbers as input and returns the average of the numbers.  Include error handling for empty lists. Comment each line of the code.",
    "Create a recipe for chocolate chip cookies.  The recipe should include specific measurements for each ingredient and detailed instructions.  Also, provide nutritional information per cookie (calories, fat, sugar)."
]

# Initialize the model
print("Initializing model...")
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

# warm up
messages = [{"role": "user", "content": "Hello, how are you?"}]
prompts = tokenizer.apply_chat_template(messages, tokenize=False)
outputs = llm.generate(prompts, sampling_params)

# Test latency
latencies = []
total_tokens = []
total_input_tokens = []
total_output_tokens = []

print(f"\nTesting latency with {num_prompts} prompts...")
for i, prompt in enumerate(test_prompts[:num_prompts]):
    messages = [{"role": "user", "content": prompt}]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Measure generation time
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    latency = end_time - start_time
    latencies.append(latency)
    
    # Count tokens
    input_tokens = len(tokenizer.encode(prompts))
    output_tokens = len(tokenizer.encode(outputs[0].outputs[0].text))
    total_tokens.append(input_tokens + output_tokens)
    total_input_tokens.append(input_tokens)
    total_output_tokens.append(output_tokens)
    
sum_latency = sum(latencies)
sum_tokens = sum(total_tokens)
tokens_per_second = sum_tokens / sum_latency

print("\nLatency Statistics:")
print(f"Total latency: {sum_latency:.2f}s")
print(f"Total tokens: {sum_tokens}")
print(f"Total input tokens: {sum(total_input_tokens)}")
print(f"Total output tokens: {sum(total_output_tokens)}")
print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")

# write to json file
with open(f"latency_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    json.dump({
        "model_id": model_id,
        "number_gpus": number_gpus,
        "num_prompts": num_prompts,
        "total_latency": sum_latency,
        "total_tokens": sum_tokens,
        "total_input_tokens": sum(total_input_tokens),
        "total_output_tokens": sum(total_output_tokens),
        "tokens_per_second": tokens_per_second
    }, f, indent=4)