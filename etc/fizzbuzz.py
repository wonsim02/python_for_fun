from itertools import cycle
from operator import itemgetter
from sys import stdout

stdout.write("\n".join(map(itemgetter(1), map(max, zip(
    zip(cycle([0]), map(str, range(1, 101))),
    zip(map(int(-1).__add__, map(int(2).__mul__, map(int(0).__eq__, map(int(3).__rmod__, range(1, 101))))), cycle(["fizz"])),
    zip(map(int(-1).__add__, map(int(2).__mul__, map(int(0).__eq__, map(int(5).__rmod__, range(1, 101))))), cycle(["buzz"])),
    zip(map(int(-2).__add__, map(int(4).__mul__, map(int(0).__eq__, map(int(15).__rmod__, range(1, 101))))), cycle(["fizzbuzz"]))
)))))
