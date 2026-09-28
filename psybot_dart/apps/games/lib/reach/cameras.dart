enum CameraFacing { front, back, external }

class CameraChoice {
  const CameraChoice({
    required this.id,
    required this.facing,
    required this.index,
  });

  final String id;
  final CameraFacing facing;
  final int index;

  bool get mirrored => facing == CameraFacing.front;

  String get label {
    switch (facing) {
      case CameraFacing.front:
        return 'front';
      case CameraFacing.back:
        return 'back';
      case CameraFacing.external:
        return 'external';
    }
  }
}

String labelFor(List<CameraChoice> all, CameraChoice one) {
  final int same =
      all.where((CameraChoice c) => c.facing == one.facing).length;
  if (same <= 1) return one.label;
  final int rank = all
          .where((CameraChoice c) => c.facing == one.facing)
          .toList()
          .indexOf(one) +
      1;
  return '${one.label} $rank';
}

int preferredIndex(List<CameraChoice> all) {
  if (all.isEmpty) return -1;
  for (int i = 0; i < all.length; i++) {
    if (all[i].facing == CameraFacing.front) return i;
  }
  return 0;
}

int nextIndex(List<CameraChoice> all, int current) {
  if (all.isEmpty) return -1;
  if (current < 0) return preferredIndex(all);
  return (current + 1) % all.length;
}

int indexOfId(List<CameraChoice> all, String? id) {
  if (id == null) return -1;
  for (int i = 0; i < all.length; i++) {
    if (all[i].id == id) return i;
  }
  return -1;
}

int resolveIndex(List<CameraChoice> all, String? savedId) {
  final int found = indexOfId(all, savedId);
  if (found >= 0) return found;
  return preferredIndex(all);
}
