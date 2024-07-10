from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# SciBERT modelini ve tokenizer'ını yükle
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Benzerlik karşılaştırılacak cümleleri tanımla
cümle1 = "aylin okula geldi."
cümle2 = "Genetik araştırmalarda devrim, DNA dizileme teknolojilerindeki ilerlemeler sayesinde başlamıştır."

# Her bir cümleyi SciBERT modeli ile temsil et
token1 = tokenizer(cümle1, return_tensors="pt")
token2 = tokenizer(cümle2, return_tensors="pt")

# Temsil edilen cümlelerin vektörlerini al
with torch.no_grad():
    output1 = model(**token1)
    output2 = model(**token2)

# Vektörlerin içeriklerini al ve ortalamasını al
vektör1 = output1.last_hidden_state.mean(dim=1).squeeze().numpy()
vektör2 = output2.last_hidden_state.mean(dim=1).squeeze().numpy()

# Kosinüs benzerliğini hesapla
dot_product = np.dot(vektör1, vektör2)
norm1 = np.linalg.norm(vektör1)
norm2 = np.linalg.norm(vektör2)
benzerlik = dot_product / (norm1 * norm2)

# Sonucu ekrana yazdır
print(f"Iki cümlenin benzerliği: {benzerlik}")


