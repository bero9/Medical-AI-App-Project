from django.urls import path
from .views import AnalyzeView , ReadTextView


urlpatterns = [
    path("analyze/", AnalyzeView.as_view(), name="analyze"),
    path("read-text/", ReadTextView.as_view(), name="read_text"),
]
