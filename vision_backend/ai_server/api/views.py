from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import ImageUploadSerializer
from .vision import analyze_image   # نموذجك

class AnalyzeView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        image = serializer.validated_data["image"]

        result = analyze_image(image)

        return Response(result, status=200)
