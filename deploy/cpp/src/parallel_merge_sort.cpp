#include <omp.h>

#include "parallel_merge_sort.h"

constexpr int max_depth = 3;

void mergeSortParallelBody(std::vector<std::pair<int, float>>& in, std::vector<std::pair<int, float>>& out, size_t start, size_t end, int depth) {
	// Base Case
	if(end <= start + 1) {
		if(in[end].second > in[start].second) {
			out[end]   = in[start];
			out[start] = in[end];
		}

		return;
	}

	// Divide path
	size_t mid = (start + end) / 2;

#pragma omp task shared(in, out) if(depth <= max_depth)

	{
		mergeSortParallelBody(in, out, start, mid, depth + 1);
	}

	{
		mergeSortParallelBody(in, out, mid + 1, end, depth + 1);
	}

#pragma omp taskwait

	// Combine path
	size_t flag       = start;
	size_t flag_left  = start;
	size_t flag_right = mid + 1;

	while(flag_left <= mid && flag_right <= end) {
		if(out[flag_left].second > out[flag_right].second) {
			in[flag++] = out[flag_left++];
		} else {
			in[flag++] = out[flag_right++];
		}
	}

	for(size_t i = flag_left; i <= mid; ++i) {
		in[flag++] = out[i];
	}

	for(size_t i = flag_right; i <= end; ++i) {
		in[flag++] = out[i];
	}

	for(size_t i = start; i <= end; ++i) {
		out[i] = in[i];
	}
}

void                 mergeSortParallel(std::vector<std::pair<int, float>>& in, std::vector<std::pair<int, float>>& out, size_t start, size_t end) {
#pragma omp parallel num_threads(4)
	{
#pragma omp single
		{
			mergeSortParallelBody(in, out, start, end, 1);
		}
	}
}

void oddEven(std::vector<std::pair<int, float>>& a, int n) {
	int                   phase;
	int                   i;
	std::pair<int, float> tmp;

#pragma omp parallel num_threads(4) default(none) shared(a, n) private(i, tmp, phase)
	for(phase = 0; phase < n; ++phase) {
		if(phase % 2 == 0) {
#pragma omp for
			for(i = 0; i < n; i += 2) {
				if(a[i - 1].second < a[i].second) {
					tmp      = a[i - 1];
					a[i - 1] = a[i];
					a[i]     = tmp;
				}
			}
		} else {
#pragma omp for
			for(i = 1; i < n - 1; i += 2) {
				if(a[i].second < a[i + 1].second) {
					tmp      = a[i + 1];
					a[i + 1] = a[i];
					a[i]     = tmp;
				}
			}
		}
	}
}

void swap(std::pair<int, float>& a, std::pair<int, float>& b) {
	std::pair<int, float> t = a;
	a                       = b;
	b                       = t;
}

int partition(std::vector<std::pair<int, float>>& arr, int start, int end) {
	std::pair<int, float> pivot = arr[end];
	int                   i     = (start - 1);

	for(int j = start; j <= end - 1; ++j) {
		if(arr[j].second < pivot.second) {
			swap(arr[i], arr[j]);
		}
	}

	swap(arr[i + 1], arr[end]);

	return (i + 1);
}

void quicksort(std::vector<std::pair<int, float>>& arr, int start, int end) {
	int index;

	if(start < end) {
		index = partition(arr, start, end);

#pragma omp parallel sections
		{
#pragma omp section
			{
				quicksort(arr, start, index - 1);
			}

#pragma omp section
			{
				quicksort(arr, index + 1, end);
			}
		}
	}
}

void merge(std::vector<std::pair<int, float>>& aux, int left, int middle, int right) {
	std::vector<std::pair<int, float>> temp(middle - left + 1);
	std::vector<std::pair<int, float>> temp2(right - middle);

	for(int i = 0; i < (middle - left + 1); ++i) {
		temp[i] = aux[left + i];
	}

	for(int i = 0; i < (right - middle); ++i) {
		temp2[i] = aux[middle + 1 + i];
	}

	int i = 0;
	int j = 0;
	int k = left;

	while(i < (middle - left + 1) && j < (right - middle)) {
		if(temp[i].second > temp2[j].second) {
			aux[k++] = temp[i++];
		} else {
			aux[k++] = temp2[j++];
		}
	}

	while(i < (middle - left + 1)) {
		aux[k++] = temp[i++];
	}

	while(j < (right - middle)) {
		aux[k++] = temp2[j++];
	}
}

void mergeSortSerial(std::vector<std::pair<int, float>>& aux, int left, int right) {
	if(left < right) {
		int middle = (left + right) / 2;

		mergeSortSerial(aux, left, middle);
		mergeSortSerial(aux, middle + 1, right);

		merge(aux, left, middle, right);
	}
}

void mergeSort(std::vector<std::pair<int, float>>& aux, int left, int right) {
	if(left < right) {
		if((right - left) > 1000) {
			int middle = (left + right) / 2;

#pragma omp task firstprivate(aux, left, middle)

			mergeSort(aux, left, middle);

#pragma omp task firstprivate(aux, middle, right)

			mergeSort(aux, middle + 1, right);

#pragma omp taskwait

			merge(aux, left, middle, right);
		} else {
			mergeSortSerial(aux, left, right);
		}
	}
}

void mergeSortSequential(std::vector<std::pair<int, float>>& in, std::vector<std::pair<int, float>>& out, size_t start, size_t end) {
	if(end <= start + 1) {
		if(in[end].second > in[start].second) {
			out[end]   = in[start];
			out[start] = in[end];
		}

		return;
	}

	size_t mid = (start + end) / 2;

	mergeSortSequential(in, out, start, mid);
	mergeSortSequential(in, out, mid + 1, end);

	size_t flag       = start;
	size_t flag_left  = start;
	size_t flag_right = mid + 1;

	while(flag_left <= mid && flag_right <= end) {
		if(out[flag_left].second > out[flag_right].second) {
			in[flag++] = out[flag_left++];
		} else {
			in[flag++] = out[flag_right++];
		}
	}

	for(size_t i = flag_left; i <= mid; ++i) {
		in[flag++] = out[i];
	}

	for(size_t i = flag_right; i <= end; ++i) {
		in[flag++] = out[i];
	}

	for(size_t i = start; i <= end; ++i) {
		out[i] = in[i];
	}
}
