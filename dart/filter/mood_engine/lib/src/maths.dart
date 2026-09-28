import 'dart:math' as math;

double clampd(double value, double low, double high) {
  if (value.isNaN) return value;
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

double mean(Iterable<double> values) {
  var total = 0.0;
  var count = 0;
  for (final value in values) {
    total += value;
    count++;
  }
  return count == 0 ? 0.0 : total / count;
}

double populationStd(Iterable<double> values) {
  final list = values.toList(growable: false);
  if (list.isEmpty) return 0.0;
  final average = mean(list);
  var total = 0.0;
  for (final value in list) {
    final delta = value - average;
    total += delta * delta;
  }
  return math.sqrt(total / list.length);
}

double peakToPeak(Iterable<double> values) {
  final list = values.toList(growable: false);
  if (list.isEmpty) return 0.0;
  var low = list.first;
  var high = list.first;
  for (final value in list) {
    if (value < low) low = value;
    if (value > high) high = value;
  }
  return high - low;
}

List<double> softmax(List<double> logits) {
  if (logits.isEmpty) return const [];
  var top = logits.first;
  for (final value in logits) {
    if (value > top) top = value;
  }
  final exponentials = <double>[
    for (final value in logits) math.exp(value - top),
  ];
  var total = 0.0;
  for (final value in exponentials) {
    total += value;
  }
  if (total <= 0) {
    return List<double>.filled(logits.length, 1.0 / logits.length);
  }
  return <double>[for (final value in exponentials) value / total];
}

List<double> normalise(List<double> values) {
  var total = 0.0;
  for (final value in values) {
    total += value;
  }
  if (total <= 0 || !total.isFinite) return values;
  return <double>[for (final value in values) value / total];
}

double maxOf(List<double> values) {
  var top = values.first;
  for (final value in values) {
    if (value > top) top = value;
  }
  return top;
}

double minOf(List<double> values) {
  var low = values.first;
  for (final value in values) {
    if (value < low) low = value;
  }
  return low;
}

int argMax(List<double> values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

double sumOf(Iterable<double> values) {
  var total = 0.0;
  for (final value in values) {
    total += value;
  }
  return total;
}

List<MapEntry<K, double>> rankedDescending<K>(Map<K, double> source) {
  final entries = source.entries.toList();
  final order = <K, int>{};
  for (var i = 0; i < entries.length; i++) {
    order[entries[i].key] = i;
  }
  entries.sort((a, b) {
    final byValue = b.value.compareTo(a.value);
    if (byValue != 0) return byValue;
    return order[a.key]!.compareTo(order[b.key]!);
  });
  return entries;
}

List<MapEntry<K, int>> rankedByCount<K>(Map<K, int> source) {
  final entries = source.entries.toList();
  final order = <K, int>{};
  for (var i = 0; i < entries.length; i++) {
    order[entries[i].key] = i;
  }
  entries.sort((a, b) {
    final byValue = b.value.compareTo(a.value);
    if (byValue != 0) return byValue;
    return order[a.key]!.compareTo(order[b.key]!);
  });
  return entries;
}

K? mostCommon<K>(Iterable<K> items) {
  final counts = <K, int>{};
  for (final item in items) {
    counts[item] = (counts[item] ?? 0) + 1;
  }
  if (counts.isEmpty) return null;
  return rankedByCount(counts).first.key;
}

int countOf<K>(Iterable<K> items, K target) {
  var total = 0;
  for (final item in items) {
    if (item == target) total++;
  }
  return total;
}

List<T> lastN<T>(List<T> values, int count) {
  if (values.length <= count) return values;
  return values.sublist(values.length - count);
}

double roundTo(double value, int digits) {
  if (!value.isFinite) return value;
  final factor = math.pow(10, digits).toDouble();
  return (value * factor).roundToDouble() / factor;
}
