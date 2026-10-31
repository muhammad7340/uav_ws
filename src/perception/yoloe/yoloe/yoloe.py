from ultralytics import YOLO
import os

# 1. Load the model from your local models folder
model = YOLO("./models/yoloe-11s-seg.pt")

# 2. Define custom classes
names = ["cardboard box", "fire hydrant", "chair", "person", "sofa", "red ball", "suitcase", "bag", "dumpster"]

# Note: The library looks for 'mobileclip_blt.ts' in the current working directory 
# or the Ultralytics weight cache. Since it's in your root folder, 
# get_text_pe should now find it without failing the zip read.
try:
    text_pe = model.get_text_pe(names)
    model.set_classes(names, text_pe)
except Exception as e:
    print(f"Error loading text model: {e}")
    print("Ensure mobileclip_blt.ts is not corrupted.")

# 3. Execute prediction
results = model.predict("./images/image6.png")
results[0].save(filename="./output_result.jpg")
# 4. Show results
results[0].show()