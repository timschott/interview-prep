
# class Solution:

def intersect(l1, l2):

    return [x for x in l1 if x in l2]
if __name__ == "__main__":
    print(intersect([1,4,100], [4,5,6,7,8,9]))