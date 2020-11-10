from itertools import permutations, product

ops = ['+', '-', '*', '/']

class tree:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    
    def __str__(self):
        return "({} {} {})".format(self.left, self.op, self.right)
    __repr__ = __str__

def get_tree_iterator(*args):
    if len(args)==1:
        yield args[0], eval(str(args[0]))
    elif len(args)==2:
        for order in set(permutations(args)):
            for op in ops:
                res = (order[0], op, order[1])
                yield tree(*res), eval("{}{}{}".format(*res))
    else:
        for n in range(1, len(args)):
            for order in set(permutations(args)):
                for l, r in product(
                    get_tree_iterator(*order[:n]),
                    get_tree_iterator(*order[n:])):
                    ltok, lval = l
                    rtok, rval = r
                    for op in ops:
                        try:
                            val = eval("{}{}{}".format(lval, op, rval))
                        except ZeroDivisionError:
                            continue
                        yield tree(ltok, op, rtok), val

if __name__=='__main__':
    target = 42
    for t, v in get_tree_iterator(1, 4, 5, 5, 5):
        if v == target:
            print(t)