#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>

using namespace std;

int main()
{
    vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    int sum_val = 0;
    double average_val = 0.0;

#pragma omp parallel for reduction(min : min_val) reduction(max : max_val) reduction(+ : sum_val) // Updates min_val with the minimum value found 
// across threads, Updates max_val with the maximum value found across threads, Computes the sum across threads, updating sum_val.
    for (int i = 0; i < nums.size(); ++i)
    {
        min_val = min(min_val, nums[i]);
        max_val = max(max_val, nums[i]);
        sum_val += nums[i];
    }

    average_val = static_cast<double>(sum_val) / nums.size();

    cout << "Minimum value: " << min_val << endl;
    cout << "Maximum value: " << max_val << endl;
    cout << "Sum: " << sum_val << endl;
    cout << "Average: " << average_val << endl;

    return 0;
}
