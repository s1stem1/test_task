def count_islands(matrix):
    if not matrix:
        return 0

    m, n = len(matrix), len(matrix[0])
    num_islands = 0

    def deep(i, j):
        if 0 <= i < m and 0 <= j < n and matrix[i][j] == 1:
            matrix[i][j] = -1
            deep(i - 1, j)
            deep(i + 1, j)
            deep(i, j - 1)
            deep(i, j + 1)
            return 1

        return 0

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                num_islands += deep(i, j)

    return num_islands


m, n = map(int, input("Enter the matrix size (MxN): ").split())
matrix = []
for i in range(m):
    row = list(map(int, input(f"Enter the elements of the {i + 1}th line, separated by a space: ").split()))
    matrix.append(row)

num_islands = count_islands(matrix)
print(f"Number of islands: {num_islands}")
