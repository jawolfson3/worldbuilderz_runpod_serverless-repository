import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
import os
from huggingface_hub import login

# Ensure HF token is passed explicitly
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN not found in environment variables")
login(token=token)

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------------------------------------------------------
# Load model + tokenizer once when the container starts
# --------------------------------------------------------
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
logging.info(f"Loading model {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,  # fixed deprecation
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

logging.info("Model loaded successfully.")

# --------------------------------------------------------
# Define your handler (runs when a job comes in)
# --------------------------------------------------------
def handler(job):
    try:
        if "input" not in job or not isinstance(job["input"], dict):
            return {"error": "Job must include an 'input' object."}

        input_data = job["input"]

        # Required generation params
        max_new_tokens = input_data.get("max_new_tokens")
        temperature = input_data.get("temperature")
        do_sample = input_data.get("do_sample", False)  # toggle, default False

        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            return {"error": "'max_new_tokens' must be a positive integer"}
        if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 1.0):
            return {"error": "'temperature' must be between 0.0 and 1.0"}

        # --- Prompt Assembly ---
        if "instruction" in input_data and "context" in input_data:
            instruction = input_data["instruction"]
            context = input_data["context"]

            if not isinstance(instruction, str) or not instruction.strip():
                return {"error": "Instruction must be a non-empty string."}
            if not isinstance(context, str) or not context.strip():
                return {"error": "Context must be a non-empty string."}

            formatted_prompt = f"<s>[INST] {instruction.strip()}\n\n{context.strip()} [/INST]"
        elif "prompt" in input_data:
            prompt = input_data["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                return {"error": "Prompt must be a non-empty string."}
            formatted_prompt = f"<s>[INST] {prompt.strip()} [/INST]"
        else:
            return {"error": "Must provide either 'prompt' or ('instruction' + 'context')."}

        # Token length validation
        tokens = tokenizer.encode(formatted_prompt)
        if len(tokens) > 30000:
            return {"error": f"Prompt too long. Max supported tokens ≈ 30000, got {len(tokens)}."}

        logging.info(
            f"Processing {len(tokens)} tokens | max_new_tokens={max_new_tokens}, "
            f"temperature={temperature}, do_sample={do_sample}"
        )

        # Run inference
        outputs = generator(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )

        response_text = outputs[0]["generated_text"]

        # Strip echoed prompt if still included
        if response_text.startswith(formatted_prompt):
            response_text = response_text[len(formatted_prompt):].strip()

        logging.info("Generation completed successfully.")
        return {"response": response_text}

    except RuntimeError:
        logging.error("GPU runtime error", exc_info=True)
        return {"error": "Internal error: contact support"}
    except Exception:
        logging.exception("Unexpected error during handler execution")
        return {"error": "Internal error: contact support"}


# --------------------------------------------------------
# Keep server alive for RunPod
# --------------------------------------------------------
runpod.serverless.start({"handler": handler})

