from django.urls import path # type: ignore
from . import views


urlpatterns = [
    path('giris', views.giris),
    path('kayit', views.kayit),
    path('giris-yap', views.giris_yap),
    path('kaydol', views.kaydol),
    path('ara', views.ara),
    
]
