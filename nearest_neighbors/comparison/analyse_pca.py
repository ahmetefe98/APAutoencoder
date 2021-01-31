import json
import statistics
from copy import deepcopy
import operator

with open('./pca32/acc.txt', 'r') as f:
    acc32 = json.load(f)
with open('./pca64/acc.txt', 'r') as f:
    acc64 = json.load(f)
with open('./pca128/acc.txt', 'r') as f:
    acc128 = json.load(f)

print('pca32:')
print('min: ', min(acc32))
print('mean: ', statistics.mean(acc32))
print('median: ', statistics.median(acc32))
print('max: ', max(acc32))

indexed = list(enumerate(acc32))
top5 = sorted(indexed, key=operator.itemgetter(1))[:5]
worst12 = sorted(indexed, key=operator.itemgetter(1))[-12:]
print('top5 class: ', top5)
print('worst12 class: ', worst12)

print('pca64:')
print('min: ', min(acc64))
print('mean: ', statistics.mean(acc64))
print('median: ', statistics.median(acc64))
print('max: ', max(acc64))

indexed = list(enumerate(acc64))
top5 = sorted(indexed, key=operator.itemgetter(1))[:5]
worst12 = sorted(indexed, key=operator.itemgetter(1))[-12:]
print('top5 class: ', top5)
print('worst12 class: ', worst12)

print('pca128:')
print('min: ', min(acc128))
print('mean: ', statistics.mean(acc128))
print('median: ', statistics.median(acc128))
print('max: ', max(acc128))

indexed = list(enumerate(acc128))
top5 = sorted(indexed, key=operator.itemgetter(1))[:5]
worst12 = sorted(indexed, key=operator.itemgetter(1))[-12:]
print('top5 class: ', top5)
print('worst12 class: ', worst12)