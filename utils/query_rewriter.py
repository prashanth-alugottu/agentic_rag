# from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# # free model from HF
# MODEL_NAME = "google/flan-t5-small" 

# print("🔄 Loading TensorFlow HF model (first time only)...")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, from_pt=True)  # TF mode

# def rewrite_query(query: str) -> str:
#     """Rewrite search query using TensorFlow HuggingFace model"""
#     prompt = f"Rewrite this query for document search: {query}"

#     input_ids = tokenizer(prompt, return_tensors="tf").input_ids
#     output_ids = model.generate(input_ids, max_length=50, num_beams=4)

#     rewritten = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return rewritten
