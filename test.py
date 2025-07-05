from transformers import AutoModel, AutoTokenizer, T5EncoderModel

checkpoint = "/data/hugang/JjyCode/llm/codet5p-220m"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = T5EncoderModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

inputs = tokenizer("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
print(inputs)
attention_mask = inputs['attention_mask']
embedding = model(**inputs)
# embedding = (embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
print(embedding.shape)