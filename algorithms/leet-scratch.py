
# class Solution:

def fizzbuzzish():
    ans = []
    for i in range(1, 101):
        
        if (i % 6 == 0):
            print(str(i) + " ab")
        elif (i % 2 == 0):
            print(str(i) + " a")
        elif (i % 3 == 0):
            print(str(i) + " b")
        else:
            print(i)

if __name__ == "__main__":
    fizzbuzzish()