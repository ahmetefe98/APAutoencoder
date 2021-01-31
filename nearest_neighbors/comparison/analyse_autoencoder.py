import json
import statistics
from copy import deepcopy
import operator

with open('./autoencoder32/acc.txt', 'r') as f:
    acc32 = json.load(f)
with open('./autoencoder64/acc.txt', 'r') as f:
    acc64 = json.load(f)
with open('./autoencoder128/acc.txt', 'r') as f:
    acc128 = json.load(f)

print('autoencoder32:')
print('mean: ', statistics.mean(acc32))
print('median: ', statistics.median(acc32))

indexed = list(enumerate(acc32))
top5 = sorted(indexed, key=operator.itemgetter(1))[:5]
worst12 = sorted(indexed, key=operator.itemgetter(1))[-12:]
print('top5 class: ', top5)
print('worst12 class: ', worst12)

print('autoencoder64:')
print('mean: ', statistics.mean(acc64))
print('median: ', statistics.median(acc64))

indexed = list(enumerate(acc64))
top5 = sorted(indexed, key=operator.itemgetter(1))[:5]
worst12 = sorted(indexed, key=operator.itemgetter(1))[-12:]
print('top5 class: ', top5)
print('worst12 class: ', worst12)

print('autoencoder128:')
print('mean: ', statistics.mean(acc128))
print('median: ', statistics.median(acc128))

indexed = list(enumerate(acc128))
top5 = sorted(indexed, key=operator.itemgetter(1))[:5]
worst12 = sorted(indexed, key=operator.itemgetter(1))[-12:]
print('top5 class: ', top5)
print('worst12 class: ', worst12)
