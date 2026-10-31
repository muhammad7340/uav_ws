# #--------- smaolvlm model with streaming :11.553-17.06 sec -----------------
# import os
# import time
# from PIL import Image
# from optimum.intel import OVModelForVisualCausalLM
# from transformers import AutoProcessor, TextIteratorStreamer
# from threading import Thread
# import openvino as ov

# # Enable caching
# ov.Core().set_property({'CACHE_DIR': './ov_cache'})

# model_path = "." 

# print("Loading model...")
# processor = AutoProcessor.from_pretrained(model_path)
# model = OVModelForVisualCausalLM.from_pretrained(
#     model_path, 
#     device="CPU", 
#     ov_config={"PERFORMANCE_HINT": "LATENCY"}
# )

# # Load image
# image = Image.open("pic.jpg").convert("RGB")
# prompt = "User:<image>\nDescribe this image in detail."
# inputs = processor(text=prompt, images=image, return_tensors="pt")

# # Streamer setup
# streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

# generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
# thread = Thread(target=model.generate, kwargs=generation_kwargs)

# print("\n--- Generating (Streaming) ---")

# # Start timers
# start_time = time.time()
# first_token_time = None

# thread.start()

# for new_text in streamer:
#     # Capture the time the very first chunk of text arrives
#     if first_token_time is None:
#         first_token_time = time.time()
    
#     print(new_text, end="", flush=True)

# thread.join()
# end_time = time.time()

# # Calculations
# ttft = first_token_time - start_time if first_token_time else 0
# total_latency = end_time - start_time

# print("\n\n" + "="*30)
# print(f"PERFORMANCE METRICS:")
# print(f"Time to First Token (TTFT): {ttft:.3f} seconds")
# print(f"Total End-to-End Latency:   {total_latency:.3f} seconds")
# print("="*30)


#--------- smaolvlm model without streaming :16.57 sec -----------------
# import os
# import time
# from PIL import Image
# from optimum.intel import OVModelForVisualCausalLM
# from transformers import AutoProcessor
# import openvino as ov

# # Enable caching
# ov.Core().set_property({'CACHE_DIR': './ov_cache'})

# model_path = "." 

# print("Loading model from local files...")
# processor = AutoProcessor.from_pretrained(model_path)
# model = OVModelForVisualCausalLM.from_pretrained(
#     model_path, 
#     device="CPU", 
#     ov_config={"PERFORMANCE_HINT": "LATENCY"}
# )

# # Load your local image
# image = Image.open("pic.jpg").convert("RGB")
# prompt = "User:<image>\nDescribe this image in detail."
# inputs = processor(text=prompt, images=image, return_tensors="pt")

# print("\n--- Generating (Non-Streaming / Blocking) ---")

# # Start timer
# start_time = time.time()

# # Generate the whole response at once
# generated_ids = model.generate(**inputs, max_new_tokens=200)

# # End timer
# end_time = time.time()

# # Decode the output
# generated_text = processor.batch_decode(
#     generated_ids[:, inputs.input_ids.shape[1]:], 
#     skip_special_tokens=True
# )

# total_latency = end_time - start_time
# # In non-streaming, the "first token" is only visible when the total latency ends
# ttft_perceived = total_latency 

# print("\n--- Model Output ---")
# print(generated_text[0])

# print("\n" + "="*35)
# print(f"NON-STREAMING PERFORMANCE:")
# print(f"Perceived TTFT:      {ttft_perceived:.3f} seconds")
# print(f"Total Latency:       {total_latency:.3f} seconds")
# print("="*35)
# print("Note: In non-streaming mode, you wait for the")
# print("entire duration before seeing a single word.")

import os
import time
from PIL import Image
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextIteratorStreamer
from threading import Thread
import openvino as ov

# Enable caching
ov.Core().set_property({'CACHE_DIR': './ov_cache'})

model_path = "." 

print("Loading SmolVLM2-256M model...")
processor = AutoProcessor.from_pretrained(model_path)
model = OVModelForVisualCausalLM.from_pretrained(
    model_path, 
    device="CPU", 
    ov_config={"PERFORMANCE_HINT": "LATENCY"}
)

# Load image
image = Image.open("image6.png").convert("RGB")

# --- THE FIX: Use the Chat Template structure ---
# This matches the {% for message in messages %} logic in the template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            #{"type": "text", "text": "how many fire hydrants can you see?"}
            #{"type": "text", "text": "describe the scene?"}
            #{"type": "text", "text": "describe only those objects in the scene about which you are 70 percent sure"}
            #{"type": "text", "text": "whats the colour of walls?"}
            {"type": "text", "text": "Describe your environment in 5 lines."}
        ]
    }
]

# 1. Generate the formatted prompt string
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# 2. Process inputs (This handles the patch-size expansion automatically)
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Streamer setup
streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

print("\n--- Generating (Streaming SmolVLM2) ---")

start_time = time.time()
first_token_time = None

thread.start()

for new_text in streamer:
    if first_token_time is None:
        first_token_time = time.time()
    print(new_text, end="", flush=True)

thread.join()
end_time = time.time()

# Calculations
ttft = first_token_time - start_time if first_token_time else 0
total_latency = end_time - start_time

print("\n\n" + "="*30)
print(f"PERFORMANCE METRICS:")
print(f"Time to First Token (TTFT): {ttft:.3f} seconds")
print(f"Total End-to-End Latency:   {total_latency:.3f} seconds")
print("="*30)



# --- Generating (Streaming SmolVLM2) ---
#  The image depicts a scene that appears to be set in what appears to be a living room or a living area. The room is characterized by a large, grey couch and a few chairs. The couch is positioned in the center of the room, with the chair to its right. The couch is a classic, rectangular piece of furniture with a dark color, possibly black or grey. The chair is a standard, rectangular piece of furniture with a light color.

# The living area is divided into two sections by a large, grey wall. The left section has a person standing, possibly a person in a casual outfit, possibly a teenager or young adult. The person is wearing a white shirt and blue jeans, and they are standing in front of the couch. The right section has a person sitting, possibly a young adult, and they are also dressed in casual attire. The person is wearing a blue shirt and blue jeans, and they are sitting on the couch.

# The floor is carpeted

# ==============================
# PERFORMANCE METRICS:
# Time to First Token (TTFT): 30.727 seconds
# Total End-to-End Latency:   45.702 seconds
# ==============================
