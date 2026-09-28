import 'dart:typed_data';

import 'package:image_picker/image_picker.dart';

import 'photo.dart';
import 'raster.dart';

Future<Raster?> pickOwnPhoto() async {
  final ImagePicker picker = ImagePicker();
  final XFile? file = await picker.pickImage(
    source: ImageSource.gallery,
    imageQuality: 92,
  );
  if (file == null) return null;
  final Uint8List bytes = await file.readAsBytes();
  return rasterFromEncodedBytes(bytes);
}
