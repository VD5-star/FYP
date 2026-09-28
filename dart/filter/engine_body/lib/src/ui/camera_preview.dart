import 'package:flutter/material.dart';

import '../platform/tracker_config.dart';

class BodyCameraPreview extends StatelessWidget {
  const BodyCameraPreview({
    required this.preview,
    required this.overlay,
    super.key,
    this.fit = BoxFit.contain,
  });

  final PreviewInfo preview;

  final Widget overlay;

  final BoxFit fit;

  @override
  Widget build(BuildContext context) {
    if (preview.width <= 0 || preview.height <= 0) {
      return const ColoredBox(color: Colors.black);
    }

    final quarterTurns = (preview.rotation ~/ 90) % 4;

    final rotated = quarterTurns.isOdd;
    final uprightWidth = rotated ? preview.height : preview.width;
    final uprightHeight = rotated ? preview.width : preview.height;

    Widget content = SizedBox(
      width: uprightWidth.toDouble(),
      height: uprightHeight.toDouble(),
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (quarterTurns == 0)
            Texture(textureId: preview.textureId)
          else
            RotatedBox(
              quarterTurns: quarterTurns,
              child: Texture(textureId: preview.textureId),
            ),
          RepaintBoundary(child: overlay),
        ],
      ),
    );

    if (preview.isMirrored) {
      content = Transform(
        alignment: Alignment.center,
        transform: Matrix4.identity()..scaleByDouble(-1.0, 1.0, 1.0, 1.0),
        child: content,
      );
    }

    return ClipRect(
      child: FittedBox(
        fit: fit,
        clipBehavior: Clip.hardEdge,
        child: content,
      ),
    );
  }
}
