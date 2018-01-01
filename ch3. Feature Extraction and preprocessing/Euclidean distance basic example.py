from sklearn.metrics.pairwise import euclidean_distances
# counts는 벡터 예제
counts = [
    [[0, 1, 1, 0, 0, 1, 0, 1]],
    [[0, 1, 1, 1, 1, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 1, 0]]
]

print('Distance between 1st and 2nd documents:', euclidean_distances(counts[0],counts[1]))
print('Distance between 1st and 3rd documents:', euclidean_distances(counts[0],counts[2]))
print('Distance between 1st and 2nd documents:', euclidean_distances(counts[1],counts[2]))
      
