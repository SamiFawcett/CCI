import numpy as np
import pandas as pd
import random
import string
import sys


def howSum(targetSum, numbers):

    if targetSum == 0:
        return []
    if targetSum < 0:
        return None

    for num in numbers:
        remainder = targetSum - num
        remainderResult = howSum(remainder, numbers)

        if remainderResult != None:
            remainderResult.append(num)
            return remainderResult

    return None


def majorityElement(k, n, arrs):

    majorityElements = []
    for i in range(k):
        freq_map = {}
        for j in range(n):
            if arrs[i][j] in freq_map:
                freq_map[arrs[i][j]] += 1
            if arrs[i][j] not in freq_map:
                freq_map[arrs[i][j]] = 1

        maxCount = 0
        val = -1
        for key in freq_map:
            if freq_map[key] > n / 2:
                maxCount = freq_map[key]
                val = key

        majorityElements.append(val)

    return majorityElements


def mtsa(arr1, arr2):
    new_arr = []
    arr1_index = 0
    arr2_index = 0
    while len(new_arr) != len(arr1) + len(arr2):
        if arr1_index == len(arr1):
            new_arr.append(arr2[arr2_index])
            arr2_index += 1
        elif arr2_index == len(arr2):
            new_arr.append(arr1[arr1_index])
            arr1_index += 1
        elif arr1[arr1_index] <= arr2[arr2_index]:
            new_arr.append(arr1[arr1_index])
            arr1_index += 1
        else:
            new_arr.append(arr2[arr2_index])
            arr2_index += 1

    return new_arr


def bestSum(targetSum, numbers, memo={}):
    if targetSum in memo:
        return memo[targetSum]
    if targetSum == 0:
        return []
    if targetSum < 0:
        return None

    smallestCombination = None

    for num in numbers:
        remainder = targetSum - num
        remainderResult = bestSum(remainder, numbers, memo)
        if remainderResult != None:
            combination = remainderResult.copy()
            combination.append(num)
            if smallestCombination == None:
                smallestCombination = combination
            elif len(combination) < len(smallestCombination):
                smallestCombination = combination

    memo[targetSum] = smallestCombination
    return smallestCombination


def canConstruct(targetWord, wordBank, memo={}):
    if targetWord in memo:
        return memo[targetWord]
    if targetWord == "":
        return True

    for word in wordBank:
        if targetWord.find(word) == 0:
            suffix = targetWord[len(word) :]
            memo[targetWord] = canConstruct(suffix, wordBank)
            if memo[targetWord]:
                return True

    return False


def countConstruct(targetWord, wordBank, memo={}):
    if targetWord in memo:
        return memo[targetWord]

    if targetWord == "":
        return 1

    total = 0
    for word in wordBank:
        if targetWord.find(word) == 0:
            suffix = targetWord[len(word) :]
            total += countConstruct(suffix, wordBank)
            memo[targetWord] = total

    return total


def allConstructs(targetWord, wordBank):
    if targetWord == "":
        return [[]]

    result = []

    for word in wordBank:
        if targetWord.find(word) == 0:
            suffix = targetWord[len(word) :]
            suffix_ways = allConstructs(suffix, wordBank)
            targetWays = []
            for way in suffix_ways:
                way.insert(0, word)
                targetWays.append(way)

            for way in targetWays:
                result.append(way)

    return result


def createRandomAdjenencyList(objects, density, directed):
    adjenencyList = {}

    if density > 1:
        density = 1

    for obj in objects:
        adjenencyList[obj] = set()

    maxConnections = np.round(len(objects) * density)
    for node in adjenencyList:
        numConnections = random.randint(0, maxConnections)
        connections = set()
        for i in range(numConnections):
            rand = objects[random.randint(0, len(objects) - 1)]
            if rand == node:
                numConnections += 1
                continue
            else:
                connections.add(rand)

        adjenencyList[node] = set(list(adjenencyList[node]) + list(connections))

        if directed == False:
            for con in list(connections):
                adjenencyList[con].add(node)

    for node in adjenencyList:
        adjenencyList[node] = list(adjenencyList[node])

    return adjenencyList


def hasPath(src, dest, graph, visited):
    if src == dest:
        return True
    if src in visited:
        return False

    visited.add(src)

    for neighbor in graph[src]:
        pathFound = hasPath(neighbor, dest, graph, visited)
        if pathFound == True:
            return True

    return False


def shortestPath(src, dest, graph):
    queue = [(src, 0)]
    visited = set()

    while len(queue) > 0:
        node = queue.pop(0)
        if node[0] in visited:
            continue
        if node[0] == dest:
            return node[1]

        visited.add(node[0])
        for neighbor in graph[node[0]]:
            queue.append((neighbor, node[1] + 1))

    return -1


def largestComponent(graph):
    largestComponentSize = 0

    visited = set()
    for node in graph:
        componentSize = getComponentSize(graph, node, visited)
        if componentSize > largestComponentSize:
            largestComponentSize = componentSize

    return largestComponentSize


def getComponentSize(graph, node, visited):
    if node in visited:
        return 0

    visited.add(node)
    count = 1

    for neighbor in graph[node]:
        count += getComponentSize(graph, neighbor, visited)

    return count


def howManyConnectedComponents(graph):
    count = 0
    visited = set()
    for node in graph:
        if dfs(graph, node, visited):
            count += 1

    return count


def dfs(graph, start, visited):
    if start in visited:
        return False
    visited.add(start)

    for neighbor in graph[start]:
        dfs(graph, neighbor, visited)

    return True


def generateIslands(r, c):

    grid = []
    for i in range(r):
        row = []
        for j in range(c):
            row.append(random.randint(0, 1))
        grid.append(np.array(row))

    return np.array(grid)


def howManyIslands(islandGrid):
    m = len(islandGrid)
    n = len(islandGrid[0])

    visited = set()
    count = 0

    for i in range(m):
        for j in range(n):
            if (i, j) in visited or islandGrid[i, j] == 0:
                continue
            else:
                exploreIsland(i, j, islandGrid, visited)
                count += 1

    return count


def exploreIsland(i, j, islandGrid, visited):
    if (i, j) in visited:
        return
    if i < 0 or j >= len(islandGrid[0]):
        return
    if j < 0 or i >= len(islandGrid):
        return
    if islandGrid[i, j] == 0:
        return

    visited.add((i, j))

    exploreIsland(i - 1, j, islandGrid, visited)
    exploreIsland(i + 1, j, islandGrid, visited)
    exploreIsland(i, j - 1, islandGrid, visited)
    exploreIsland(i, j + 1, islandGrid, visited)


def minimumIslandSize(islandGrid):
    m = len(islandGrid)
    n = len(islandGrid[0])

    visited = set()

    minimumIslandSize = sys.maxsize
    for i in range(m):
        for j in range(n):
            if (i, j) not in visited and islandGrid[i, j] == 1:
                foundIslandSize = islandSize(i, j, islandGrid, visited)
                if foundIslandSize < minimumIslandSize:
                    minimumIslandSize = foundIslandSize

    if minimumIslandSize == sys.maxsize:
        return -1
    else:
        return minimumIslandSize


def islandSize(i, j, island, visited):
    if (i, j) in visited:
        return 0

    if i < 0 or j >= len(island[0]):
        return 0

    if j < 0 or i >= len(island):
        return 0

    if island[i, j] == 0:
        return 0

    visited.add((i, j))

    return (
        islandSize(i - 1, j, island, visited)
        + islandSize(i + 1, j, island, visited)
        + islandSize(i, j - 1, island, visited)
        + islandSize(i, j + 1, island, visited)
        + 1
    )


def twoSum(arrs):
    out_file = open("out.txt", "w+")
    for arr in arrs:
        map = {}
        for i in range(0, len(arr)):
            val = arr[i]
            for j in range(i + 1, len(arr)):
                if arr[j] + val == 0:
                    map[0] = (i + 1, j + 1)
        if 0 in map:
            x, y = map[0]
            st = str(x) + " " + str(y) + "\n"
            out_file.write(st)
        else:
            neg1 = -1
            st = str(neg1) + "\n"
            out_file.write(st)


def edgeListToGraph(edgelist, num_nodes, directed=False):
    adjenencyMap = {}
    for i in range(1, num_nodes + 1):
        adjenencyMap[i] = []

    for edge in edgelist:
        vertex1 = edge[0]
        vertex2 = edge[1]
        adjenencyMap[vertex1].append(vertex2)
        if not directed:
            adjenencyMap[vertex2].append(vertex1)

    return adjenencyMap


def findAllPathLengthsFromVertex1(graph, num_nodes):
    visited = set()
    queue = [(1, 0)]
    pathLengths = [0] * num_nodes

    while len(queue) > 0:
        queue_obj = queue.pop(0)
        node = queue_obj[0]
        dist = queue_obj[1]

        if node not in visited:
            pathLengths[node - 1] = dist

            visited.add(node)

            for neighbor in graph[node]:
                queue.append((neighbor, dist + 1))

    for i in range(1, num_nodes + 1):
        if i not in visited:
            pathLengths[i - 1] = -1

    return pathLengths


def rosalindConnectedComponents(edgelist, num_nodes):
    def dfs(node, graph, visited):
        if not node or node in visited:
            return False

        visited.add(node)

        for neighbor in graph[node]:
            dfs(neighbor, graph, visited)

        return True

    graph = edgeListToGraph(edgelist, num_nodes)

    visited = set()
    count = 0
    for node in graph:
        if dfs(node, graph, visited):
            count += 1

    return count


class MaxHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        self.heap.append(val)
        j = len(self.heap)
        while j - 1 != 0 and self.heap[(j // 2) - 1] < self.heap[j - 1]:
            temp = self.heap[(j // 2) - 1]
            self.heap[(j // 2) - 1] = self.heap[j - 1]
            self.heap[j - 1] = temp
            j = j // 2


# Sorting Algorithms


def mergeSortDriver(arr):
    return mergeSort(arr)


def mergeSort(arr):

    if len(arr) > 1:
        middle = len(arr) // 2

        left = arr[:middle]
        right = arr[middle:]

        mergeSort(left)
        mergeSort(right)

        arr = merge(arr, left, right)
        return arr


def merge(arr, L, R):
    # Copy data to temp arrays L[] and R[]
    i = j = k = 0
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Checking if any element was left
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1

    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1

    return arr


def twoWayPartition(arr):

    lowerStack = []
    upperStack = []

    partition = arr[0]

    for i in range(1, len(arr)):
        num = arr[i]
        if num <= partition:
            lowerStack.append(num)
        else:
            upperStack.append(num)

    assembled = []

    for num in lowerStack:
        assembled.append(num)

    assembled.append(partition)

    for num in upperStack:
        assembled.append(num)

    return assembled


def threeSum(arrs):
    # brute force solution
    """
    rets = []
    for arr in arrs:
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                for k in range(j+1, len(arr)):
                    if(arr[i] + arr[j] + arr[k] == 0):
                        rets.append([i+1, j+1, k+1])
    return rets
    """

    # optimal solution

    rets = []
    for arr in arrs:
        # 1-indexed
        initalPositions = {}
        for i in range(len(arr)):
            initalPositions[arr[i]] = i + 1
        found = 0
        # sort arr
        sortedArr = sorted(arr)
        for i in range(len(sortedArr) - 3):
            start = i + 1
            end = len(sortedArr) - 1
            while start < end and found == 0:
                if sortedArr[i] + sortedArr[start] + sortedArr[end] == 0:
                    found = 1
                    rets.append(
                        sorted(
                            [
                                initalPositions[sortedArr[i]],
                                initalPositions[sortedArr[start]],
                                initalPositions[sortedArr[end]],
                            ]
                        )
                    )
                    break
                elif sortedArr[i] + sortedArr[start] + sortedArr[end] < 0:
                    currentStart = start
                    while sortedArr[start] == sortedArr[currentStart] and start < end:
                        start += 1
                elif sortedArr[i] + sortedArr[start] + sortedArr[end] > 0:
                    currentEnd = end
                    while sortedArr[end] == sortedArr[currentEnd] and start < end:
                        end -= 1
        if found == 0:
            rets.append([-1])
    return rets


# byte-by-byte coding problems
# comment mistakes when going from paper to ide

# 1. Median of Arrays
def findMedianArrays(arr1, arr2):
    i = j = 0
    arr3 = []
    while i < len(arr1) or j < len(arr2):  # used wrong variable name for i iterator
        if i >= len(arr1):  # keep control structure parentheses consistent
            arr3.append(arr2[j])
            j += 1  # forgot to increment j
        elif j >= len(arr2):  # keep control structure parenthese consistent
            arr3.append(arr1[i])
            i += 1  # forgot to increment i
        elif arr1[i] <= arr2[j]:  # changed control structure from if to elif
            arr3.append(arr1[i])
            i += 1
        elif arr2[j] < arr1[i]:
            arr3.append(arr2[j])
            j += 1

    middle = len(arr3) // 2
    if len(arr3) % 2 == 0:
        return (arr3[middle] + arr3[middle - 1]) / 2
    elif len(arr3) % 2 == 1:
        return arr3[middle]


# Optimized Median of Arrays Solution
def findMedian(arr1, arr2):

    m1 = median(arr1)
    m2 = median(arr2)
    print(arr1, arr2)
    if len(arr1) == 2:

        if m1 > m2:
            return median([min(arr1), max(arr2)])
        else:
            return median([min(arr2), max(arr1)])

    if m1 == m2:
        return m1

    middle = len(arr1) // 2

    if m1 < m2:
        new_a1 = arr1[middle:]
        new_a2 = arr2[: middle + 1]
        return findMedian(new_a1, new_a2)
    elif m2 < m1:
        new_a1 = arr1[: middle + 1]
        new_a2 = arr2[middle1:]
        return findMedian(new_a1, new_a2)


def median(arr):
    middle = len(arr) // 2
    if len(arr) % 2 == 0:
        return (arr[middle - 1] + arr[middle]) / 2
    return arr[middle]


def isBipartite(graph):
    stack = []
    v1 = set()
    v2 = set()
    for node in graph:
        stack.append((node, 0))
        break

    while len(stack) > 0:
        node, vs = stack.pop(len(stack) - 1)

        if vs == 0:
            if node in v1:
                continue
            elif node in v2:
                return -1
            v1.add(node)
        elif vs == 1:
            if node in v2:
                continue
            elif node in v1:
                return -1
            v2.add(node)

        for neighbor in graph[node]:
            stack.append((neighbor, (vs + 1) % 2))

    return 1


def knapsack(arr, maxWeight):
    # sort value/weight pair by decreasing values
    arr.sort(reverse=True, key=lambda x: x[1])
    totalWeight = 0
    totalVal = 0
    i = 0
    while totalWeight < maxWeight and i < len(arr):

        pair = arr[i]
        p_weight = pair[0]
        p_val = pair[1]
        if p_weight + totalWeight <= maxWeight:
            totalWeight += p_weight
            totalVal += p_val
        i += 1
    return (totalWeight, totalVal)


def knapsack(profits, weights, total_weight, n, memo={}):
    if n in memo:
        return memo[n]
    if n == -1 or total_weight < 0:
        return 0

    if weights[n] > total_weight:
        memo[n] = knapsack(profits, weights, total_weight, n - 1)
        return memo[n]
    elif weights[n] <= total_weight:
        memo[n] = max(
            profits[n] + knapsack(profits, weights, total_weight - weights[n], n - 1),
            knapsack(profits, weights, total_weight, n - 1),
        )
        return memo[n]


if __name__ == "__main__":

    # Graph Algorithms
    objects = ["a", "b", "c", "d", "e"]
    g = createRandomAdjenencyList(objects, 0.5, False)
    print(g)

    print(hasPath("a", "d", g, set()))
    print(howManyConnectedComponents(g))
    print(largestComponent(g))
    print(shortestPath("a", "d", g))

    islandGrid = generateIslands(5, 5)

    print(islandGrid)
    print(howManyIslands(islandGrid))

    # ROSALIND PROBLEM

    # BYTE BY BYTE PROBLEM
    # median = findMedian([1, 3, 5], [2, 4, 6])
    # print(median)

    # Recursive Knapsack
    profits = [6, 10, 12]
    weights = [1, 2, 3]
    max_weight = 5
    n = 2
    solution = knapsack(profits, weights, max_weight, n)
    print(solution)
