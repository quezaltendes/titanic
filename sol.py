import os
import io
import re
import json
import pickle
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# =============================================================================
# 🔧 Настройки
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./Qwen3-VL-8B-Instruct-FP8"

# =============================================================================
# 🔹 Загрузка модели и процессора
# =============================================================================
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# =============================================================================
# 🔹 Преобразования изображений (из vllm)
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=1536, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((int(target_width), int(target_height)))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# =============================================================================
# 🔹 Инференс
# =============================================================================


def extract_answer_from_response(response: str) -> str:
    """
    Извлекает ответ из последней строки модели:
    - если есть двоеточие — берёт всё после него (в верхнем регистре);
    - если нет — берёт последний элемент после split().
    Возвращает только буквы A–F (уникальные, в порядке появления).
    Если результат не соответствует ожидаемым вариантам (A, B, C, D, AB, AC, и т.д.), возвращает "A".
    """
    if not response:
        return "A"  # Возвращаем A по умолчанию для пустого ответа

    last_line = response.strip().splitlines()[-1].strip()

    if ':' in last_line:
        part = last_line.split(':', 1)[1].strip().upper()
    else:
        part = last_line.split()[-1].strip().upper() if last_line.split() else ""

    # Извлекаем только буквы A-F
    matches = re.findall(r'[A-F]', part)
    if matches:
        seen = set()
        result = ''.join([m for m in matches if not (m in seen or seen.add(m))])
        
        # Проверяем, соответствует ли результат ожидаемым вариантам
        if is_valid_answer(result):
            return result
        else:
            return "A"
    
    return "A"  # Возвращаем A, если не найдено подходящих букв

def is_valid_answer(answer: str) -> bool:
    """
    Проверяет, является ли ответ допустимым:
    - Должен содержать только буквы A, B, C, D (можно расширить до F при необходимости)
    - Длина от 1 до 2 символов (можно настроить)
    - Буквы должны быть уникальными
    """
    if not answer or len(answer) > 2:  # Максимум 2 буквы
        return False
    
    # Проверяем, что все символы находятся в диапазоне A-D
    if not all(char in 'ABCD' for char in answer):
        return False
    
    # Проверяем уникальность символов (уже гарантировано в extract_answer_from_response)
    return len(set(answer)) == len(answer)


def infer_one(model, processor, image_bytes, question_text):
    """Инференс одного примера (один рисунок и вопрос)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    sub_images = dynamic_preprocess(image, image_size=1300, use_thumbnail=True)

    # Формируем prompt
    content = [{"type": "image", "image": img} for img in sub_images]
    content.append({
    "type": "text",
    "text": f"""
You are an expert assistant for solving school-level math and physics diagram questions.
Your task is to analyze the given image(s) and determine which statements (A–F) are TRUE.

Each question contains several statements labeled with A–F.
One or more of them may be correct.

Guidelines:
- Carefully examine geometric or physical relations in the image.
- Ignore any text outside the image.
- Think briefly and logically, but don't think too long.
- The FINAL ANSWER must be **only** the correct capital letters (A–F) without any spaces, commas, or extra words.
- If none are correct, answer "A" by default.
- Example of valid final line: BD

Question:
{question_text}

Answer:
"""
})

    

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=sub_images,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    # Генерация
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=3500,
            do_sample=False,
            temperature=0.0,
            top_k=1,
            pad_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    input_length = inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_length:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    answer = extract_answer_from_response(response)
    if not answer:
        answer = "AB"  # fallback, чтобы не было пустого ответа

    # 🔹 ЛОГИРОВАНИЕ
    print(f"{'='*80}")
    print(f"❓ ВОПРОС:\n{question_text}")
    print(f"{'-'*80}")
    print(f"🤖 ПОЛНЫЙ ОТВЕТ МОДЕЛИ:\n{response}")
    print(f"{'-'*80}")
    print(f"✅ ИЗВЛЕЧЕННЫЙ ОТВЕТ: {answer}")
    print(f"{'='*80}\n")

    return answer


  

# =============================================================================
# 🔹 Основная функция (чтение input.pickle и запись output.json)
# =============================================================================

def main():
    input_path = "input.pickle"
    output_path = "output.json"

    # Читаем входные данные
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    results = []
    for item in data:
        rid = item.get("rid")
        question = item.get("question", "")
        image_bytes = item.get("image", None)

        if image_bytes is None:
            print(f"⚠️ RID {rid} has no image, skipping.")
            continue

        answer = infer_one(model, processor, image_bytes, question)
        results.append({"rid": rid, "answer": answer})

    # Сохраняем результат
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Results saved to {output_path}")

if __name__ == "__main__":
    main()
