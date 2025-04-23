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
]

# Initialize the model
print("Initializing model...")
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

# Test latency
latencies = []
total_tokens = []

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
    
sum_latency = sum(latencies)
sum_tokens = sum(total_tokens)
tokens_per_second = sum_tokens / sum_latency

print("\nLatency Statistics:")
print(f"Total latency: {sum_latency:.2f}s")
print(f"Total tokens: {sum_tokens:.2f}")
print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")

# write to json file
with open(f"latency_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    json.dump({
        "model_id": model_id,
        "number_gpus": number_gpus,
        "num_prompts": num_prompts,
        "total_latency": sum_latency,
        "total_tokens": sum_tokens,
        "tokens_per_second": tokens_per_second
    }, f)