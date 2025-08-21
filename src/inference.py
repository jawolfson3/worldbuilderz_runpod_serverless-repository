import runpod

# Define your handler (this runs whenever a job comes in)
def handler(job):
    prompt = job["input"].get("prompt", "No prompt provided.")
    return {"response": f"Echo: {prompt}"}

# Tell RunPod to keep the server alive and wait for jobs
runpod.serverless.start({"handler": handler})


