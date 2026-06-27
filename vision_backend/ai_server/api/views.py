from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
 # استدعاء الدالة المخصصة
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
# لا تنسَ استدعاء الدالة الجديدة في أعلى الملف
from .read_text import extract_text


class ReadTextView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):

        serializer = ImageUploadSerializer(data=request.data)

        if serializer.is_valid():

            image = serializer.validated_data["image"]

            result = extract_text(image)


            return Response(
                {
                    "text": result,
                    "tts_text": result
                },
                status=status.HTTP_200_OK
            )


        print("❌ Serializer Error:", serializer.errors)

        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )