import numpy as np
import pandas as pd
import random
import string
import sys


def howSum(targetSum, numbers):

    if(targetSum == 0):
        return []
    if(targetSum < 0):
        return None

    for num in numbers:
        remainder = targetSum - num
        remainderResult = howSum(remainder, numbers)

        if(remainderResult != None):
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
            if(freq_map[key] > n/2):
                maxCount = freq_map[key]
                val = key

        majorityElements.append(val)

    return majorityElements


def mtsa(arr1, arr2):
    new_arr = []
    arr1_index = 0
    arr2_index = 0
    while(len(new_arr) != len(arr1) + len(arr2)):
        if(arr1_index == len(arr1)):
            new_arr.append(arr2[arr2_index])
            arr2_index += 1
        elif(arr2_index == len(arr2)):
            new_arr.append(arr1[arr1_index])
            arr1_index += 1
        elif(arr1[arr1_index] <= arr2[arr2_index]):
            new_arr.append(arr1[arr1_index])
            arr1_index += 1
        else:
            new_arr.append(arr2[arr2_index])
            arr2_index += 1

    return new_arr


def bestSum(targetSum, numbers, memo={}):
    if(targetSum in memo):
        return memo[targetSum]
    if(targetSum == 0):
        return []
    if(targetSum < 0):
        return None

    smallestCombination = None

    for num in numbers:
        remainder = targetSum - num
        remainderResult = bestSum(remainder, numbers, memo)
        if(remainderResult != None):
            combination = remainderResult.copy()
            combination.append(num)
            if(smallestCombination == None):
                smallestCombination = combination
            elif(len(combination) < len(smallestCombination)):
                smallestCombination = combination

    memo[targetSum] = smallestCombination
    return smallestCombination


def canConstruct(targetWord, wordBank, memo={}):
    if(targetWord in memo):
        return memo[targetWord]
    if(targetWord == ''):
        return True

    for word in wordBank:
        if(targetWord.find(word) == 0):
            suffix = targetWord[len(word):]
            memo[targetWord] = canConstruct(suffix, wordBank)
            if(memo[targetWord]):
                return True

    return False


def countConstruct(targetWord, wordBank, memo={}):
    if targetWord in memo:
        return memo[targetWord]

    if targetWord == '':
        return 1

    total = 0
    for word in wordBank:
        if(targetWord.find(word) == 0):
            suffix = targetWord[len(word):]
            total += countConstruct(suffix, wordBank)
            memo[targetWord] = total

    return total


def allConstructs(targetWord, wordBank):
    if targetWord == '':
        return [[]]

    result = []

    for word in wordBank:
        if(targetWord.find(word) == 0):
            suffix = targetWord[len(word):]
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

    if(density > 1):
        density = 1

    for obj in objects:
        adjenencyList[obj] = set()

    maxConnections = np.round(len(objects) * density)
    for node in adjenencyList:
        numConnections = random.randint(0, maxConnections)
        connections = set()
        for i in range(numConnections):
            rand = objects[random.randint(0, len(objects)-1)]
            if rand == node:
                numConnections += 1
                continue
            else:
                connections.add(rand)

        adjenencyList[node] = set(
            list(adjenencyList[node]) + list(connections))

        if(directed == False):
            for con in list(connections):
                adjenencyList[con].add(node)

    for node in adjenencyList:
        adjenencyList[node] = list(adjenencyList[node])

    return adjenencyList


def hasPath(src, dest, graph, visited):
    if src == dest:
        return True
    if(src in visited):
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

    while (len(queue) > 0):
        node = queue.pop(0)
        if node[0] in visited:
            continue
        if node[0] == dest:
            return node[1]

        visited.add(node[0])
        for neighbor in graph[node[0]]:
            queue.append((neighbor, node[1]+1))

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
        if(dfs(graph, node, visited)):
            count += 1

    return count


def dfs(graph, start, visited):
    if(start in visited):
        return False
    visited.add(start)

    for neighbor in graph[start]:
        dfs(graph, neighbor, visited)

    return True


def twoSum(arrs):

    return


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
            if((i, j) in visited or islandGrid[i, j] == 0):
                continue
            else:
                exploreIsland(i, j, islandGrid, visited)
                count += 1

    return count


def exploreIsland(i, j, islandGrid, visited):
    if ((i, j) in visited):
        return
    if(i < 0 or j >= len(islandGrid[0])):
        return
    if(j < 0 or i >= len(islandGrid)):
        return
    if(islandGrid[i, j] == 0):
        return

    visited.add((i, j))

    exploreIsland(i-1, j, islandGrid, visited)
    exploreIsland(i+1, j, islandGrid, visited)
    exploreIsland(i, j-1, islandGrid, visited)
    exploreIsland(i, j+1, islandGrid, visited)


def minimumIslandSize(islandGrid):
    m = len(islandGrid)
    n = len(islandGrid[0])

    visited = set()

    minimumIslandSize = sys.maxsize
    for i in range(m):
        for j in range(n):
            if (i, j) not in visited and islandGrid[i, j] == 1:
                foundIslandSize = islandSize(i, j, islandGrid, visited)
                if(foundIslandSize < minimumIslandSize):
                    minimumIslandSize = foundIslandSize

    if minimumIslandSize == sys.maxsize:
        return -1
    else:
        return minimumIslandSize


def islandSize(i, j, island, visited):
    if((i, j) in visited):
        return 0

    if(i < 0 or j >= len(island[0])):
        return 0

    if(j < 0 or i >= len(island)):
        return 0

    if(island[i, j] == 0):
        return 0

    visited.add((i, j))

    return islandSize(i-1, j, island, visited) + islandSize(i+1, j, island, visited) + islandSize(i, j-1, island, visited) + islandSize(i, j+1, island, visited) + 1


if __name__ == '__main__':

    objects = ['a', 'b', 'c', 'd', 'e']
    g = createRandomAdjenencyList(objects, .5, False)
    print(g)

    print(hasPath('a', 'd', g, set()))
    print(howManyConnectedComponents(g))
    print(largestComponent(g))
    print(shortestPath('a', 'd', g))

    islandGrid = generateIslands(5, 5)

    print(islandGrid)
    print(howManyIslands(islandGrid))
    print(minimumIslandSize(islandGrid))

'''
    file = open('file.txt', 'r')
    lines = file.readlines()
    arrs = []
    for line in lines:
        arr = [int(i) for i in line.split(' ')]
        arrs.append(arr)
    indicies = twoSum(arrs)
    for indexRange in indicies:
        if(indexRange != -1):
            print(indexRange[0], indexRange[1])
        else:
            print(indexRange)
'''
