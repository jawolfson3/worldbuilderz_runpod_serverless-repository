import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

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
    torch_dtype=torch.bfloat16,
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
        # Validate job input
        if "input" not in job or not isinstance(job["input"], dict):
            logging.warning("Job missing 'input' object.")
            return {"error": "Job must include an 'input' object."}

        input_data = job["input"]

        # Prompt validation
        prompt = input_data.get("prompt", None)
        if prompt is None:
            logging.warning("No prompt provided in job.")
            return {"error": "No prompt provided."}
        if not isinstance(prompt, str):
            logging.warning(f"Invalid prompt type: {type(prompt)}")
            return {"error": "Prompt must be a string."}
        if not prompt.strip():
            logging.warning("Prompt was empty.")
            return {"error": "Prompt cannot be empty."}

        # Token length validation
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 30000:
            logging.warning(f"Prompt too long: {len(tokens)} tokens.")
            return {
                "error": f"Prompt too long. Max supported tokens ≈ 30000, received {len(tokens)}."
            }

        # Required params from input
        if "max_new_tokens" not in input_data:
            logging.warning("Missing 'max_new_tokens' in input.")
            return {"error": "Missing required field: max_new_tokens"}
        if "temperature" not in input_data:
            logging.warning("Missing 'temperature' in input.")
            return {"error": "Missing required field: temperature"}

        max_new_tokens = input_data["max_new_tokens"]
        temperature = input_data["temperature"]

        # Validate parameter types
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            return {"error": "'max_new_tokens' must be a positive integer"}
        if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 1.0):
            return {"error": "'temperature' must be a number between 0.0 and 1.0"}

        logging.info(
            f"Processing prompt of {len(tokens)} tokens with max_new_tokens={max_new_tokens}, temperature={temperature}"
        )

        # Run inference
        outputs = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )

        response_text = outputs[0]["generated_text"]
        logging.info("Generation completed successfully.")
        return {"response": response_text}

    except RuntimeError as e:
        # GPU-related errors (like out of memory)
        logging.error("GPU runtime error", exc_info=True)
        return {"error": "Internal error: contact support"}
    except Exception as e:
        # Unexpected errors
        logging.exception("Unexpected error during handler execution")
        return {"error": "Internal error: contact support"}

# --------------------------------------------------------
# Keep server alive for RunPod
# --------------------------------------------------------
runpod.serverless.start({"handler": handler})
