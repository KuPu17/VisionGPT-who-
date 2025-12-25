import torch
import cv2
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import settings

class VLMInference:
    def __init__(self):
        print(f"Loading VLM ({settings.QWEN_MODEL_NAME})... this takes RAM!")
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                settings.QWEN_MODEL_NAME, 
                torch_dtype=torch.float32, 
                device_map=settings.DEVICE
            )
            self.processor = AutoProcessor.from_pretrained(settings.QWEN_MODEL_NAME)
            print("✓ VLM Loaded.")
        except Exception as e:
            print(f"✗ Critical Error loading VLM: {e}")
            raise e

    def ask(self, frame_bgr, prompt_text):
        print("\n🤖 Qwen is thinking...")
        
        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to configured device
        inputs = inputs.to(settings.DEVICE)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"💡 Answer: {response}\n")
        return response