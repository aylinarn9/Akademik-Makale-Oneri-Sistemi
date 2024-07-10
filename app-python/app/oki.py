import os

# Okunan dosyaların verilerini depolamak için boş bir liste oluşturun
all_data = []

# Metin dosyalarının bulunduğu klasörün yolu
klasor_yolu = 'C:/Users/Aylin/Downloads/Inspec/Inspec/docsutf8'

# Klasördeki tüm dosya isimlerini alfabetik sırayla alın
dosya_isimleri = sorted(os.listdir(klasor_yolu))

# Alınan dosya isimleri üzerinde bir döngü oluşturun
for dosya in dosya_isimleri:
    if dosya.endswith('.txt'):  # Sadece .txt dosyalarını işle
        dosya_yolu = os.path.join(klasor_yolu, dosya)  # Dosya yolunu oluştur
        with open(dosya_yolu, 'r') as file:  # Dosyayı aç
            # Dosyadaki veriyi oku ve all_data listesine ekle
            data = file.read()
            all_data.append(data)  # Dosya verisini ekle

# Tüm dosyalar alfabetik sırayla okunacak şekilde all_data listesinde saklanacak

#%%
import fasttext # type: ignore
import numpy as np # type: ignore

model=fasttext.load_model("C:/Users/Aylin/Downloads/cc.en.300.bin/cc.en.300.bin")


import re
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

# Örnek makale metinleri listesi


# Metni temizleme ve tokenizasyon
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldırma
    text = text.lower()  # Küçük harfe dönüştürme
    return text

def tokenize_text(text):
    tokens = word_tokenize(text)  # Metni kelime parçalarına ayırma
    return [token for token in tokens if token not in stopwords.words('english')]  # Stop kelimeleri kaldırma (opsiyonel)


OnİslenmisMakale=[]
MakaleVektorleri=[]
for i in all_data:
    cleaned_text = clean_text(i)
    x = cleaned_text.replace("\n", "")
    y=tokenize_text(x)
    z = " ".join(y)
    OnİslenmisMakale.append(z)
    vektor=model.get_sentence_vector(z)
    MakaleVektorleri.append(vektor)
#%%
def OneriBul(ilgiAlanlari):

    ilgi=model.get_sentence_vector(" ".join(tokenize_text(clean_text(ilgiAlanlari))))

    benzerlikler=[]
    for i in MakaleVektorleri:
        similarity = np.dot(i,ilgi) / (np.linalg.norm(i) * np.linalg.norm(ilgi))
        benzerlikler.append(similarity)

    en_buyukler = sorted(range(len(benzerlikler)), key=lambda i: benzerlikler[i], reverse=True)[:5]
    print(en_buyukler)

    Oneriler=[]
    for i in en_buyukler:
        Oneriler.append(OnİslenmisMakale[i])
        
    return Oneriler 




from transformers import AutoTokenizer, AutoModel
import torch

# SciBERT modelini ve tokenizer'ını yükle
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

diziyeni=[]
for i in OnİslenmisMakale:
    token1 = tokenizer(i, return_tensors="pt")
    with torch.no_grad():
        output1 = model(**token1)
    vektör1 = output1.last_hidden_state.mean(dim=1).squeeze().numpy()
    diziyeni.append(vektör1)   


def Oneribul2(ilgiAlanlari):
    # Benzerlik karşılaştırılacak cümleleri tanımla
    token1 = tokenizer(ilgiAlanlari, return_tensors="pt")
    with torch.no_grad():
        output1 = model(**token1)
    ilgialanvek = output1.last_hidden_state.mean(dim=1).squeeze().numpy()


    benzerlikler2=[]
    for j in diziyeni:
        dot_product = np.dot(j,ilgialanvek)
        norm1 = np.linalg.norm(j)
        norm2 = np.linalg.norm( ilgialanvek)
        benzerlik = dot_product / (norm1 * norm2) 
        benzerlikler2.append(benzerlik)

    en_buyukler2 = sorted(range(len(benzerlikler2)), key=lambda i: benzerlikler2[i], reverse=True)[:5]
    print(en_buyukler2)

    Oneriler2=[]
    for i in en_buyukler2:
        Oneriler2.append(OnİslenmisMakale[i])
        
    return Oneriler2 


for i in Oneribul2("education"):
    print(i)






