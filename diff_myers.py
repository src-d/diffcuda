#!/usr/bin/env python3

import sys

def restore(archive, D, x, y):
    for V in archive:
        print(*V)

def diff(t1, t2):
    N = len(t1)
    M = len(t2)
    MAX = M + N
    zp = MAX
    state = [(0,)] * (2 * MAX + 1)
    archive = []

    for D in range(MAX + 1):
        for k in range(-D, D + 1, 2):
            if k == -D or (k != D and state[k-1 + zp] < state[k+1 + zp]):
                x = state[k + 1 + zp][0]
                ref = "up"
            else:
                x = state[k - 1 + zp][0] + 1
                ref = "left"
            y = x - k
            print(x, y, t1[x], t2[y])
            while x < N and y < M and t1[x] == t2[y]:
                x += 1
                y += 1
            state[k + zp] = x, ref
            print(k, x)
            if x >= N and y >= M:
                rpath = []
                while D > 0:
                    if x <= N and y <= M:
                        while x > 0 and y > 0 and t1[x - 1] == t2[y - 1]:
                            x -= 1
                            y -= 1
                    if x == 0 and y == 0:
                        break
                    ref = state[k + zp][1]
                    if ref == "up":
                        y -= 1
                        rpath.append("%d %d + %s" % (k, x + 1, t2[y]))
                        k += 1
                    else:
                        x -= 1
                        rpath.append("%d %d - %s" % (k, x + 1, t1[x]))
                        k -= 1
                    D -= 1
                    state = archive[D]
                return reversed(rpath)
        archive.append(list(state))

with open(sys.argv[1]) as f1:
    with open(sys.argv[2]) as f2:
        tdiff = diff([l[:-1] for l in f1.readlines()],
                     [l[:-1] for l in f2.readlines()])

for l in tdiff:
    print(l)
