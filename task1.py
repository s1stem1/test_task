N = int(input('Enter positive integer number:'))
summa = 0
if N <= 0:
    print('You need to enter a positive integer')
else:
    for i in range(1, N + 1):
        summa += i
    print(summa)
