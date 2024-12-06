from PIL import Image
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
from time import sleep

# Configure the Gemini Model API
genai.configure(api_key="YOUR_API_KEY")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Artifacts
categories = {
    "airplane": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Implausible aerodynamic structures",
        "Misaligned body panels",
        "Impossible mechanical joints",
        "Distorted window reflections",
    ],
    "automobile": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Incorrect wheel geometry",
        "Misaligned body panels",
        "Impossible mechanical joints",
        "Distorted window reflections",
    ],
    "ship": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Misaligned body panels",
    ],
    "truck": [
        "Artificial noise patterns in uniform surfaces",
        "Metallic surface artifacts",
        "Impossible mechanical connections",
        "Inconsistent scale of mechanical parts",
        "Physically impossible structural elements",
        "Incorrect wheel geometry",
        "Misaligned body panels",
        "Impossible mechanical joints",
        "Distorted window reflections",
    ],
    "bird": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
    ],
    "cat": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Anatomically incorrect paw structures",
        "Improper fur direction flows",
    ],
    "deer": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Improper fur direction flows",
    ],
    "dog": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Dental anomalies in mammals",
        "Anatomically incorrect paw structures",
        "Improper fur direction flows",
    ],
    "frog": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
    ],
    "horse": [
        "Unrealistic eye reflections",
        "Misshapen ears or appendages",
        "Anatomically impossible joint configurations",
        "Unnatural pose artifacts",
        "Biological asymmetry errors",
        "Regular grid-like artifacts in textures",
        "Impossible foreshortening in animal bodies",
        "Misaligned bilateral elements in animal faces",
        "Over-smoothing of natural textures",
        "Dental anomalies in mammals",
    ],
    "major": [
        "Discontinuous surfaces",
        "Non-manifold geometries in rigid structures",
        "Asymmetric features in naturally symmetric objects",
        "Texture bleeding between adjacent regions",
        "Excessive sharpness in certain image regions",
        "Artificial smoothness",
        "Movie-poster-like composition of ordinary scenes",
        "Unnatural lighting gradients",
        "Fake depth of field",
        "Abruptly cut-off objects",
        "Color coherence breaks",
        "Spatial relationship errors",
        "Depth perception anomalies",
        "Over-sharpening artifacts",
        "Incorrect reflection mapping",
        "Inconsistent object boundaries",
        "Floating or disconnected components",
        "Texture repetition patterns",
        "Unrealistic specular highlights",
        "Inconsistent material properties",
        "Inconsistent shadow directions",
        "Multiple light source conflicts",
        "Missing ambient occlusion",
        "Incorrect perspective rendering",
        "Scale inconsistencies within single objects",
        "Aliasing along high-contrast edges",
        "Blurred boundaries in fine details",
        "Jagged edges in curved structures",
        "Random noise patterns in detailed areas",
        "Loss of fine detail in complex structures",
        "Artificial enhancement artifacts",
        "Repeated element patterns",
        "Systematic color distribution anomalies",
        "Frequency domain signatures",
        "Unnatural color transitions",
        "Resolution inconsistencies within regions",
        "Glow or light bleed around object boundaries",
        "Ghosting effects: Semi-transparent duplicates of elements",
        "Cinematization effects",
        "Dramatic lighting that defies natural physics",
        "Artificial depth of field in object presentation",
        "Unnaturally glossy surfaces",
        "Synthetic material appearance",
        "Multiple inconsistent shadow sources",
        "Exaggerated characteristic features",
        "Scale inconsistencies within the same object class",
        "Incorrect skin tones",
    ],
}

prompt = """Analyze the provided image, and its corresponding Grad-CAM output which has been resized to 32x32. Focus primarily on the original image to identify and explain distinguishing artifacts that indicate it is fake. Use the Grad-CAM output for reference only when necessary. Provide clear, concise explanations (maximum 50 words each) using the specified artifacts below. Include positional references like 'top left' or 'bottom right' when relevant. DO NOT include any other sentences or artifacts in your response. Select only 6-7 relevant artifacts.
Write each artifact and explanation on a separate line, using the format:
Artifact Name: Explanation.
For example:
Unrealistic eye reflections: Unnatural symmetrical light reflections in both eyes, suggesting generated elements.
Over-smoothing of natural textures: Fur appears unusually smooth in the top right, lacking natural texture variation.

Notes:
Explanations should remain under 50 words for clarity.
DO NOT reference artifacts not listed or include extra commentary.

ONLY use the artifacts listed below:
"""

file_name = "output.csv"
print("Reading", file_name)

df = pd.read_csv(file_name)

original_img_path = "task2_images"
gradcam_img_path = "gradcam_images"

for i, row in tqdm(df.iterrows(), total=len(df)):
    if not pd.isna(row["response"]):
        continue

    image_path = row["image_path"]
    cl = row["label"]
    text = (
        prompt
        + "["
        + ", ".join(categories["Major Issues"])
        + ", ".join(categories[cl])
        + "]"
    )
    img1 = Image.open(f"{original_img_path}/{image_path}").convert("RGB")
    img2 = Image.open(f"{gradcam_img_path}/{image_path}").convert("RGB")
    response = model.generate_content([prompt, img1, img2])
    # print(text)
    # print(response.text)
    # break
    df.loc[i, "response"] = response.text
    sleep(4)
    if i > 0 and i % 5 == 0:
        df.to_csv(file_name, index=False)

df.to_csv(file_name, index=False)
