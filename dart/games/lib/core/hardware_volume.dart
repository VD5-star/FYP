import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

typedef VolumeCallback = void Function(int delta);

class HardwareVolume extends StatefulWidget {
  const HardwareVolume({
    super.key,
    required this.child,
    required this.onVolume,
  });

  final Widget child;
  final VolumeCallback onVolume;

  @override
  State<HardwareVolume> createState() => _HardwareVolumeState();
}

class _HardwareVolumeState extends State<HardwareVolume> {
  @override
  void initState() {
    super.initState();
    HardwareKeyboard.instance.addHandler(_handle);
  }

  @override
  void dispose() {
    HardwareKeyboard.instance.removeHandler(_handle);
    super.dispose();
  }

  bool _handle(KeyEvent e) {
    if (e is! KeyDownEvent) return false;
    final LogicalKeyboardKey k = e.logicalKey;
    if (k == LogicalKeyboardKey.audioVolumeUp) {
      widget.onVolume(5);
      return true;
    }
    if (k == LogicalKeyboardKey.audioVolumeDown) {
      widget.onVolume(-5);
      return true;
    }
    return false;
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      autofocus: true,
      onKeyEvent: (FocusNode n, KeyEvent e) {
        if (e is KeyDownEvent) {
          if (e.logicalKey == LogicalKeyboardKey.audioVolumeUp) {
            widget.onVolume(5);
            return KeyEventResult.handled;
          }
          if (e.logicalKey == LogicalKeyboardKey.audioVolumeDown) {
            widget.onVolume(-5);
            return KeyEventResult.handled;
          }
        }
        return KeyEventResult.ignored;
      },
      child: widget.child,
    );
  }
}

mixin HardwareVolumeMixin<T extends StatefulWidget> on State<T> {
  int _vol = 55;

  int get appVolume => _vol;

  void bindVolume(int v) => _vol = v.clamp(0, 100);

  void changeVolume(int delta, {void Function(int v)? apply}) {
    _vol = (_vol + delta).clamp(0, 100);
    if (apply != null) apply(_vol);
    if (mounted) setState(() {});
  }

  Widget wrapVolume(Widget child, void Function(int v) onChange) {
    return HardwareVolume(
      onVolume: (int d) => changeVolume(d, apply: onChange),
      child: child,
    );
  }
}
