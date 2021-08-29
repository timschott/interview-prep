from typing import List

## The objective with this file is to have a flexible place to drop/run
## leet code problems locally. It is not meant to be a master catalogue
## of each solution.

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for first_index, item in enumerate(nums):
            for second_index, item in enumerate(nums):
                if (first_index != second_index):
                    if nums[first_index] + nums[second_index] == target:
                        return [first_index, second_index]
        
        else:
            return 'No target found'

foo = Solution()
print(foo.twoSum([3,2,4], 6))