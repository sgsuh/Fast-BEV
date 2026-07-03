#pragma once

#include <vector>

void mergeSortParallel(std::vector<std::pair<int, float>>& in, std::vector<std::pair<int, float>>& out, size_t start, size_t end);
void oddEven(std::vector<std::pair<int, float>>& a, int n);
void quicksort(std::vector<std::pair<int, float>>& arr, int start, int end);
void mergeSort(std::vector<std::pair<int, float>>& aux, int left, int right);
void mergeSortSequential(std::vector<std::pair<int, float>>& in, std::vector<std::pair<int, float>>& out, size_t start, size_t end);
