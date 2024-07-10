from django.shortcuts import render
from .models import Person
from . import oki
def giris(request):
    return render(request, 'giris.html')

def kayit(request):
    return render(request, 'kayit.html')

def giris_yap(request):
    if request.method == 'POST':
        
        eposta = request.POST.get('eposta')
        sifre = request.POST.get('sifre')
        sonuc=Person.find_one({"eposta": eposta, "sifre": sifre})
        
        if sonuc:
            print(sonuc["ilgi"])
            return render(request, 'app.html', {'data': oki.OneriBul(sonuc["ilgi"])}) 
        else:
            return render(request, 'giris.html')



def ara(request):
    if request.method == 'POST':
        
        arama = request.POST.get('arama')
        

        if arama:
            return render(request, 'app.html', {'data': oki.OneriBul(arama)}) 
        else:
            return render(request, 'app.html')


def kaydol(request):
    if request.method == 'POST':
        
        eposta = request.POST.get('eposta')
        sifre = request.POST.get('sifre')
        ilgi = request.POST.get('ilgi')

        üye={
            "eposta":eposta,
            "sifre":sifre,
            "ilgi":ilgi
        }
        Person.insert_one(üye)
        return render(request, 'kayit.html') 