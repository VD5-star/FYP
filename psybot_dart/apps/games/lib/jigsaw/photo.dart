import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'raster.dart';

Future<Raster> rasterFromEncodedBytes(Uint8List bytes) async {
  final ui.Codec codec = await ui.instantiateImageCodec(bytes);
  final ui.FrameInfo frame = await codec.getNextFrame();
  final ui.Image image = frame.image;
  final int w = image.width;
  final int h = image.height;
  final ByteData? data = await image.toByteData(
    format: ui.ImageByteFormat.rawRgba,
  );
  image.dispose();
  if (data == null) {
    throw StateError('could not read decoded image pixels');
  }
  return Raster(w, h, data.buffer.asUint8List());
}

Raster fitRaster(Raster image, int maxW, int maxH) {
  final double s = math.min(maxW / image.w, maxH / image.h);
  final int nw = math.max(1, (image.w * s).round());
  final int nh = math.max(1, (image.h * s).round());
  return resizeRaster(image, nw, nh);
}

Raster snapToGrid(Raster image, int rows, int cols) {
  final int nw = (image.w ~/ cols) * cols;
  final int nh = (image.h ~/ rows) * rows;
  if (nw == image.w && nh == image.h) return image;
  return cropRaster(image, (image.w - nw) ~/ 2, (image.h - nh) ~/ 2, nw, nh);
}
