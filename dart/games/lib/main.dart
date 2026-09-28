import 'package:flutter/material.dart';

import 'attend/game.dart';
import 'core/adaptive.dart';
import 'jigsaw/game.dart';
import 'reach/game.dart';

void main() {
  runApp(const PsybotApp());
}

class PsybotApp extends StatelessWidget {
  const PsybotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'psybot games',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF171614),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  void _open(BuildContext context, Widget game) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (BuildContext c) => Scaffold(
          backgroundColor: const Color(0xFF171614),
          body: SafeArea(child: game),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final List<List<Object>> games = <List<Object>>[
      <Object>['reach', 'move your body to the marks', 0],
      <Object>['jigsaw', 'put the picture back together', 1],
      <Object>['attend', 'five sounds, one at a time', 2],
    ];
    final double s = Adaptive.scale(context);
    final double maxW = Adaptive.isTablet(context) ? 600 : 520;
    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (BuildContext context, BoxConstraints outer) {
            return SingleChildScrollView(
              child: ConstrainedBox(
                constraints: BoxConstraints(minHeight: outer.maxHeight),
                child: Center(
                  child: ConstrainedBox(
                    constraints: BoxConstraints(maxWidth: maxW),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: <Widget>[
                        Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Text(
                            'psybot',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: Adaptive.font(context, 40),
                              color: const Color(0xFFD8DEE2),
                            ),
                          ),
                        ),
                        Padding(
                          padding: EdgeInsets.only(bottom: 32 * s.clamp(0.7, 1.1)),
                          child: Text(
                            'nothing is scored, nothing is kept',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: Adaptive.font(context, 14),
                              color: const Color(0xFF676C70),
                            ),
                          ),
                        ),
                        for (final List<Object> g in games)
                          Padding(
                            padding: EdgeInsets.symmetric(
                              horizontal: Adaptive.isSmallPhone(context) ? 16 : 24,
                              vertical: 8 * s.clamp(0.85, 1.0),
                            ),
                            child: _Tile(
                              title: g[0] as String,
                              detail: g[1] as String,
                              onTap: () {
                                final int which = g[2] as int;
                                if (which == 0) {
                                  _open(context, const ReachGame());
                                } else if (which == 1) {
                                  _open(context, const JigsawGame());
                                } else {
                                  _open(context, const AttendGame());
                                }
                              },
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class _Tile extends StatelessWidget {
  const _Tile({required this.title, required this.detail, required this.onTap});

  final String title;
  final String detail;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: const Color(0xFF32363A),
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                title,
                style: const TextStyle(fontSize: 22, color: Color(0xFFD8DEE2)),
              ),
              const SizedBox(height: 4),
              Text(
                detail,
                style: const TextStyle(fontSize: 13, color: Color(0xFF9CA3A8)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
