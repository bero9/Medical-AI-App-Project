import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:http_parser/http_parser.dart';

class BlindPage extends StatefulWidget {
  const BlindPage({super.key});

  @override
  State<BlindPage> createState() => _BlindPageState();
}

class _BlindPageState extends State<BlindPage> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isCameraReady = false;
  bool _isProcessing = false;
  final FlutterTts _flutterTts = FlutterTts();

  /// استخدم رابط ngrok أو الـ IP المحلي
  final String apiUrl = "https://f855-80-79-6-86.ngrok-free.app/api/analyze/";

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  Future<void> _initCamera() async {
    await Permission.camera.request();

    if (await Permission.camera.isDenied) {
      await _speak("عذرًا، لا يمكن فتح الكاميرا بدون إذن.");
      return;
    }

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      await _speak("لم يتم العثور على كاميرا على هذا الجهاز.");
      return;
    }

    _controller = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;

      setState(() => _isCameraReady = true);

      await _speak("تم تشغيل الكاميرا. المس الشاشة لالتقاط الصورة.");
    } catch (e) {
      await _speak("حدث خطأ أثناء تشغيل الكاميرا.");
    }
  }

  Future<void> _captureAndDescribe() async {
    if (_controller == null || !_controller!.value.isInitialized || _isProcessing) return;

    setState(() => _isProcessing = true);
    await _speak("جاري التقاط الصورة...");

    try {
      final picture = await _controller!.takePicture();
      final result = await sendImageToServer(File(picture.path));

      if (result != null && result["tts_text"] != null) {
        await _speak(result["tts_text"]);
      } else {
        await _speak("لم أتمكن من التعرف على محتوى الصورة.");
      }

    } catch (e) {
      print("⚠️ ERROR while capture: $e");
      await _speak("حدث خطأ أثناء التقاط أو تحليل الصورة.");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<Map<String, dynamic>?> sendImageToServer(File imageFile) async {
    try {
      print("📤 Sending image to API: $apiUrl");

      var uri = Uri.parse(apiUrl);
      var request = http.MultipartRequest("POST", uri);

      request.headers['Accept'] = 'application/json';

      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
          filename: "captured.jpg", // 🔥 مهم جداً
          contentType: MediaType('image', 'jpeg'),
        ),
      );

      var response = await request.send();
      var body = await response.stream.bytesToString();

      print("📥 STATUS: ${response.statusCode}");
      print("📦 BODY: $body");

      if (response.statusCode == 200) {
        return jsonDecode(body);
      }

      return null;

    } catch (e) {
      print("🔥 SEND ERROR: $e");
      return null;
    }
  }

  Future<void> _speak(String text) async {
    await _flutterTts.setLanguage("ar");
    await _flutterTts.setPitch(1.0);
    await _flutterTts.setSpeechRate(0.9);
    await _flutterTts.speak(text);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _flutterTts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: _isCameraReady
          ? GestureDetector(
        onTap: _captureAndDescribe,
        child: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(_controller!),
            if (_isProcessing)
              Container(
                color: Colors.black45,
                child: const Center(
                  child: CircularProgressIndicator(color: Colors.white),
                ),
              ),
          ],
        ),
      )
          : const Center(
        child: CircularProgressIndicator(color: Colors.white),
      ),
    );
  }
}
