import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration

model_name = 'llava-hf/llava-1.5-7b-hf'
torch_dtype = torch.float16
bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype)
processor = AutoProcessor.from_pretrained(
                model_name,
                padding_side="left", # need to have padding side as left
                device_map="auto")
model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                # quantization_config=bnb_config,
                pad_token_id=processor.tokenizer.pad_token_id,
                torch_dtype = torch_dtype,
                device_map="auto")



# prompt  = "USER: <image>Detect letter E in the image. Return coorindates in x_min,y_min,x_max, and y_max.\nASSISTANT: "
# image_path = 'rgb_test.png'
prompt  = "USER: <image>\nDetect donut in the image. Return coorindates in x_min,y_min,x_max, and y_max.\nASSISTANT: "
image_path = 'test_images/rgb.png'
image_o = Image.open(image_path)
# pad the image to be square with white background
image = Image.new("RGB", (max(image_o.size), max(image_o.size)), (255, 255, 255))
# paste the original image in the center
image.paste(image_o, (int((max(image_o.size) - image_o.size[0]) / 2), int((max(image_o.size) - image_o.size[1]) / 2)))
print(image.size)
plt.imshow(image)


# image = image.resize((224, 224))
image = image.convert("RGB")
print(image.size)

inputs = processor(prompt, image, return_tensors="pt")
for k, v in inputs.items():
    inputs[k] = v.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(outputs)

decoded_prompt = processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
decoded_output = processor.decode(outputs[0], skip_special_tokens=True)
print(decoded_prompt)
print(decoded_output)

# get the output by removing the prompt
output = decoded_output[len(decoded_prompt):]
print(output)

# get the coordinates in format x_min,y_min,x_max,y_max with float
if '\n' in output:
    multi_bbox = output.split('\n')
else:
    multi_bbox = [output]

# get height and width of the image
width, height = image.size
# draw the rectangle on the image
from PIL import ImageDraw
draw = ImageDraw.Draw(image)
for bbox in multi_bbox:
    if bbox == '':
        continue
    if '[' in bbox:
        bbox = bbox.split('[')[1].split(']')[0]
    print('bbox string: ', bbox)
    coordinates = [float(x) for x in bbox.split(',')[:4]]
    print(coordinates)
    # convert the coordinates to pixel values
    coordinates = [int(x * width) if i % 2 == 0 else int(x * height) for i, x in enumerate(coordinates)]
    print(coordinates)
    draw.rectangle(coordinates, outline="red", width=3)
plt.imshow(image)
