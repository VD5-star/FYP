import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:psybot_games/main.dart';

void main() {
  testWidgets('the home screen lists the three games', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const PsybotApp());
    await tester.pump();

    expect(find.text('psybot'), findsOneWidget);
    expect(find.text('reach'), findsOneWidget);
    expect(find.text('jigsaw'), findsOneWidget);
    expect(find.text('attend'), findsOneWidget);
    expect(find.text('nothing is scored, nothing is kept'), findsOneWidget);
  });

  testWidgets('each game opens without crashing', (
    WidgetTester tester,
  ) async {
    for (final String name in <String>['jigsaw', 'attend']) {
      await tester.pumpWidget(const PsybotApp());
      await tester.pump();
      await tester.tap(find.text(name));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 40));
      expect(tester.takeException(), isNull, reason: name);
    }
  });

  testWidgets('the home screen fits a short landscape phone', (
    WidgetTester tester,
  ) async {
    tester.view.physicalSize = const Size(800, 400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const PsybotApp());
    await tester.pump();
    expect(tester.takeException(), isNull);
  });

  testWidgets('the home screen fits a small phone', (
    WidgetTester tester,
  ) async {
    tester.view.physicalSize = const Size(320, 480);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const PsybotApp());
    await tester.pump();
    expect(tester.takeException(), isNull);
  });

  testWidgets('the home screen fits with large accessibility text', (
    WidgetTester tester,
  ) async {
    tester.view.physicalSize = const Size(400, 700);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      MediaQuery(
        data: const MediaQueryData(textScaler: TextScaler.linear(1.6)),
        child: const PsybotApp(),
      ),
    );
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}
