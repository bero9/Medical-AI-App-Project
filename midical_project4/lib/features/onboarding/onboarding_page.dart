import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _controller = PageController(
    viewportFraction: 1, // ← يمنع ظهور الصور الجانبية
  );

  int _page = 0;

  final _pages = <_OnboardPageData>[
    _OnboardPageData(
      title: 'لمحة عن الفريق الطبي',
      subtitle:
      'فريق متعدد التخصصات يضم أطباء وتمريضًا وأخصائيي تخاطب وعلاج سمعي وبصري للعمل معًا على رعاية شاملة.',
      image: "assets/images/team.jpg",
    ),
    _OnboardPageData(
      title: 'وظائف التطبيق لمرضى الصم',
      subtitle:
      'تحويل الكلام المحيط إلى نص على الشاشة، تنبيهات اهتزازية، ودعم الترجمة الفورية للنصوص الهامة.',
      image: "assets/images/deaf2.jpg",
    ),
    _OnboardPageData(
      title: 'وظائف التطبيق لمرضى البكم',
      subtitle:
      'تحويل النص إلى كلام، عبارات سريعة للتواصل الفوري، ولوحة تواصل بصري تسهّل التعبير.',
      image: "assets/images/mute.jpg",
    ),
    _OnboardPageData(
      title: 'وظائف التطبيق للمكفوفين',
      subtitle:
      'كاميرا ناطقة تصف المشهد من حولك، قراءة النصوص المطبوعة، وتوجيه صوتي للمساعدة في التنقل.',
      image: "assets/images/blind2.jpg",
    ),
  ];

  Future<void> _finishOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('seen_onboarding', true);
    if (!mounted) return;
    Navigator.of(context).pushReplacementNamed('/login');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          /// الخلفية السوداء
          Container(color: Colors.black),

          /// PageView
          Directionality(
            textDirection: TextDirection.ltr,
            child: PageView.builder(
              controller: _controller,
              itemCount: _pages.length,
              onPageChanged: (i) => setState(() => _page = i),
              itemBuilder: (context, i) {
                return Padding(
                  padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(25),
                    child: _OnboardPage(data: _pages[i]),
                  ),
                );
              },
            ),
          ),

          /// زر تخطي
          Positioned(
            top: 40,
            right: 40,
            child: InkWell(
              onTap: _finishOnboarding,
              child: SafeArea(
                child: Text(
                  "Skip",
                  style: TextStyle(
                    fontSize: 18,
                    color: Colors.blue.shade900,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),

          /// نقاط الصفحات
          Positioned(
            bottom: 20,
            left: 0,
            right: 0,
            child: Center(
              child: Directionality(
                textDirection: TextDirection.ltr,
                child: SmoothPageIndicator(
                  controller: _controller,
                  count: _pages.length,
                  effect: const WormEffect(
                    dotHeight: 10,
                    dotWidth: 10,
                    spacing: 8,
                    dotColor: Colors.white54,
                    activeDotColor: Colors.white,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _OnboardPageData {
  final String title;
  final String subtitle;
  final String image;

  const _OnboardPageData({
    required this.title,
    required this.subtitle,
    required this.image,
  });
}

class _OnboardPage extends StatelessWidget {
  final _OnboardPageData data;

  const _OnboardPage({required this.data});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        /// الصورة الخلفية (ملء الشاشة داخل البطاقات)
        SizedBox.expand(
          child: Image.asset(
            data.image,
            fit: BoxFit.cover,
          ),
        ),

        /// صندوق النصوص في الأسفل
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            padding: const EdgeInsets.all(18),
            margin: const EdgeInsets.only(bottom: 20),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.55),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  data.title,
                  textAlign: TextAlign.center,
                  textDirection: TextDirection.rtl,
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  data.subtitle,
                  textAlign: TextAlign.center,
                  textDirection: TextDirection.rtl,
                  style: const TextStyle(
                    fontSize: 16,
                    height: 1.5,
                    color: Colors.white70,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
