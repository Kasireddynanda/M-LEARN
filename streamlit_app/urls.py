from django.urls import path
from .views import upload_dataset, train_model,upload_view

urlpatterns = [
    path('upload/', upload_dataset, name='upload_dataset'),
    path('train/', train_model, name='train_model'),
      path('upload/', upload_view, name='streamlitapp_upload'),
]