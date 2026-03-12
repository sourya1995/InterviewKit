import java.util.*;
import java.util.stream.*;

/**
 * Comprehensive Coding Interview Patterns Reference - Java Edition
 * 
 * This file contains Java implementations of 85+ coding patterns
 * commonly seen in technical interviews.
 */

public class CodingPatterns {

    // ==================== ARRAY & STRING PATTERNS ====================
    
    /**
     * Pattern 1: Prefix Sum / Cumulative Sum
     * Time: O(n) preprocessing, O(1) query
     * Space: O(n)
     */
    static class PrefixSum {
        // 1D Prefix Sum
        public static int[] buildPrefixSum(int[] arr) {
            int n = arr.length;
            int[] prefix = new int[n + 1];
            for (int i = 0; i < n; i++) {
                prefix[i + 1] = prefix[i] + arr[i];
            }
            return prefix;
        }
        
        public static int rangeSum(int[] prefix, int i, int j) {
            return prefix[j + 1] - prefix[i];
        }
        
        // 2D Prefix Sum
        public static int[][] buildPrefixSum2D(int[][] matrix) {
            int rows = matrix.length;
            int cols = matrix[0].length;
            int[][] prefix = new int[rows + 1][cols + 1];
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    prefix[i+1][j+1] = matrix[i][j] + prefix[i][j+1] + 
                                      prefix[i+1][j] - prefix[i][j];
                }
            }
            return prefix;
        }
        
        public static int rangeSum2D(int[][] prefix, int r1, int c1, int r2, int c2) {
            return prefix[r2+1][c2+1] - prefix[r1][c2+1] - 
                   prefix[r2+1][c1] + prefix[r1][c1];
        }
    }
    
    /**
     * Pattern 2: Difference Array
     * Time: O(n + q) where q is number of updates
     * Space: O(n)
     */
    static class DifferenceArray {
        public static int[] applyRangeUpdates(int n, int[][] updates) {
            int[] diff = new int[n + 1];
            
            // Apply all range updates
            for (int[] update : updates) {
                int L = update[0], R = update[1], val = update[2];
                diff[L] += val;
                diff[R + 1] -= val;
            }
            
            // Reconstruct array
            int[] result = new int[n];
            result[0] = diff[0];
            for (int i = 1; i < n; i++) {
                result[i] = result[i-1] + diff[i];
            }
            return result;
        }
    }
    
    /**
     * Pattern 3: Kadane's Algorithm
     * Time: O(n)
     * Space: O(1)
     */
    static class Kadane {
        public static int maxSubArray(int[] arr) {
            int maxSum = Integer.MIN_VALUE;
            int currentSum = 0;
            
            for (int num : arr) {
                currentSum = Math.max(num, currentSum + num);
                maxSum = Math.max(maxSum, currentSum);
            }
            return maxSum;
        }
        
        // Maximum Product Subarray variation
        public static int maxProduct(int[] arr) {
            int maxProd = arr[0];
            int minProd = arr[0];
            int result = arr[0];
            
            for (int i = 1; i < arr.length; i++) {
                if (arr[i] < 0) {
                    int temp = maxProd;
                    maxProd = minProd;
                    minProd = temp;
                }
                maxProd = Math.max(arr[i], maxProd * arr[i]);
                minProd = Math.min(arr[i], minProd * arr[i]);
                result = Math.max(result, maxProd);
            }
            return result;
        }
    }
    
    /**
     * Pattern 4: Dutch National Flag (Three-Way Partitioning)
     * Time: O(n)
     * Space: O(1)
     */
    static class DutchNationalFlag {
        public static void sortColors(int[] arr) {
            int low = 0, mid = 0, high = arr.length - 1;
            
            while (mid <= high) {
                if (arr[mid] == 0) {
                    swap(arr, low, mid);
                    low++;
                    mid++;
                } else if (arr[mid] == 1) {
                    mid++;
                } else {
                    swap(arr, mid, high);
                    high--;
                }
            }
        }
        
        private static void swap(int[] arr, int i, int j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    
    /**
     * Pattern 5: Cyclic Sort
     * Time: O(n)
     * Space: O(1)
     */
    static class CyclicSort {
        public static void cyclicSort(int[] arr) {
            int i = 0;
            while (i < arr.length) {
                int correctPos = arr[i] - 1;
                if (arr[i] != arr[correctPos]) {
                    swap(arr, i, correctPos);
                } else {
                    i++;
                }
            }
        }
        
        // Find missing number
        public static int findMissingNumber(int[] arr) {
            int i = 0;
            int n = arr.length;
            while (i < n) {
                if (arr[i] < n && arr[i] != arr[arr[i]]) {
                    swap(arr, i, arr[i]);
                } else {
                    i++;
                }
            }
            
            for (i = 0; i < n; i++) {
                if (arr[i] != i) return i;
            }
            return n;
        }
        
        private static void swap(int[] arr, int i, int j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    
    /**
     * Pattern 6: Coordinate Compression
     * Time: O(n log n)
     * Space: O(n)
     */
    static class CoordinateCompression {
        public static int[] compress(int[] arr) {
            int[] sorted = arr.clone();
            Arrays.sort(sorted);
            
            Map<Integer, Integer> compressed = new HashMap<>();
            int idx = 0;
            for (int val : sorted) {
                if (!compressed.containsKey(val)) {
                    compressed.put(val, idx++);
                }
            }
            
            int[] result = new int[arr.length];
            for (int i = 0; i < arr.length; i++) {
                result[i] = compressed.get(arr[i]);
            }
            return result;
        }
    }
    
    // ==================== TWO POINTERS & SLIDING WINDOW ====================
    
    /**
     * Pattern 8: Two Pointers (Classic)
     * Time: O(n)
     * Space: O(1)
     */
    static class TwoPointers {
        // Two Sum in sorted array
        public static int[] twoSum(int[] arr, int target) {
            int left = 0, right = arr.length - 1;
            
            while (left < right) {
                int sum = arr[left] + arr[right];
                if (sum == target) {
                    return new int[]{left, right};
                } else if (sum < target) {
                    left++;
                } else {
                    right--;
                }
            }
            return new int[]{-1, -1};
        }
        
        // Container with most water
        public static int maxArea(int[] height) {
            int left = 0, right = height.length - 1;
            int maxArea = 0;
            
            while (left < right) {
                int area = Math.min(height[left], height[right]) * (right - left);
                maxArea = Math.max(maxArea, area);
                
                if (height[left] < height[right]) {
                    left++;
                } else {
                    right--;
                }
            }
            return maxArea;
        }
    }
    
    /**
     * Pattern 9: Fast & Slow Pointers
     * Time: O(n)
     * Space: O(1)
     */
    static class FastSlowPointers {
        static class ListNode {
            int val;
            ListNode next;
            ListNode(int val) { this.val = val; }
        }
        
        // Detect cycle
        public static boolean hasCycle(ListNode head) {
            ListNode slow = head, fast = head;
            
            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
                if (slow == fast) return true;
            }
            return false;
        }
        
        // Find cycle start
        public static ListNode detectCycle(ListNode head) {
            ListNode slow = head, fast = head;
            
            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
                if (slow == fast) break;
            }
            
            if (fast == null || fast.next == null) return null;
            
            slow = head;
            while (slow != fast) {
                slow = slow.next;
                fast = fast.next;
            }
            return slow;
        }
    }
    
    /**
     * Pattern 10: Sliding Window (Fixed Size)
     * Time: O(n)
     * Space: O(1)
     */
    static class SlidingWindowFixed {
        public static int maxSumSubarray(int[] arr, int k) {
            int windowSum = 0;
            for (int i = 0; i < k; i++) {
                windowSum += arr[i];
            }
            
            int maxSum = windowSum;
            for (int i = k; i < arr.length; i++) {
                windowSum += arr[i] - arr[i - k];
                maxSum = Math.max(maxSum, windowSum);
            }
            return maxSum;
        }
    }
    
    /**
     * Pattern 11: Sliding Window (Variable Size)
     * Time: O(n)
     * Space: O(k) for character set
     */
    static class SlidingWindowVariable {
        // Longest substring without repeating characters
        public static int lengthOfLongestSubstring(String s) {
            Map<Character, Integer> map = new HashMap<>();
            int maxLen = 0, left = 0;
            
            for (int right = 0; right < s.length(); right++) {
                char c = s.charAt(right);
                if (map.containsKey(c)) {
                    left = Math.max(left, map.get(c) + 1);
                }
                map.put(c, right);
                maxLen = Math.max(maxLen, right - left + 1);
            }
            return maxLen;
        }
        
        // Minimum window substring
        public static String minWindow(String s, String t) {
            if (s.length() < t.length()) return "";
            
            Map<Character, Integer> need = new HashMap<>();
            Map<Character, Integer> window = new HashMap<>();
            
            for (char c : t.toCharArray()) {
                need.put(c, need.getOrDefault(c, 0) + 1);
            }
            
            int left = 0, right = 0;
            int valid = 0;
            int start = 0, minLen = Integer.MAX_VALUE;
            
            while (right < s.length()) {
                char c = s.charAt(right);
                right++;
                
                if (need.containsKey(c)) {
                    window.put(c, window.getOrDefault(c, 0) + 1);
                    if (window.get(c).equals(need.get(c))) {
                        valid++;
                    }
                }
                
                while (valid == need.size()) {
                    if (right - left < minLen) {
                        start = left;
                        minLen = right - left;
                    }
                    
                    char d = s.charAt(left);
                    left++;
                    
                    if (need.containsKey(d)) {
                        if (window.get(d).equals(need.get(d))) {
                            valid--;
                        }
                        window.put(d, window.get(d) - 1);
                    }
                }
            }
            
            return minLen == Integer.MAX_VALUE ? "" : s.substring(start, start + minLen);
        }
    }
    
    /**
     * Pattern 12: Sliding Window with Deque
     * Time: O(n)
     * Space: O(k)
     */
    static class SlidingWindowDeque {
        public static int[] maxSlidingWindow(int[] nums, int k) {
            Deque<Integer> deque = new ArrayDeque<>();
            int[] result = new int[nums.length - k + 1];
            
            for (int i = 0; i < nums.length; i++) {
                // Remove elements outside window
                while (!deque.isEmpty() && deque.peekFirst() < i - k + 1) {
                    deque.pollFirst();
                }
                
                // Maintain decreasing order
                while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                    deque.pollLast();
                }
                
                deque.offerLast(i);
                
                if (i >= k - 1) {
                    result[i - k + 1] = nums[deque.peekFirst()];
                }
            }
            return result;
        }
    }
    
    // ==================== LINKED LIST PATTERNS ====================
    
    /**
     * Pattern 13: In-Place Linked List Manipulation
     * Time: O(n)
     * Space: O(1)
     */
    static class LinkedListManipulation {
        static class ListNode {
            int val;
            ListNode next;
            ListNode(int val) { this.val = val; }
        }
        
        // Reverse linked list
        public static ListNode reverse(ListNode head) {
            ListNode prev = null;
            ListNode curr = head;
            
            while (curr != null) {
                ListNode nextNode = curr.next;
                curr.next = prev;
                prev = curr;
                curr = nextNode;
            }
            return prev;
        }
        
        // Reverse between positions
        public static ListNode reverseBetween(ListNode head, int left, int right) {
            if (head == null || left == right) return head;
            
            ListNode dummy = new ListNode(0);
            dummy.next = head;
            ListNode prev = dummy;
            
            for (int i = 0; i < left - 1; i++) {
                prev = prev.next;
            }
            
            ListNode curr = prev.next;
            for (int i = 0; i < right - left; i++) {
                ListNode next = curr.next;
                curr.next = next.next;
                next.next = prev.next;
                prev.next = next;
            }
            
            return dummy.next;
        }
    }
    
    // ==================== STACK & QUEUE PATTERNS ====================
    
    /**
     * Pattern 15: Monotonic Stack
     * Time: O(n)
     * Space: O(n)
     */
    static class MonotonicStack {
        // Next greater element
        public static int[] nextGreaterElement(int[] arr) {
            int n = arr.length;
            int[] result = new int[n];
            Arrays.fill(result, -1);
            Stack<Integer> stack = new Stack<>();
            
            for (int i = 0; i < n; i++) {
                while (!stack.isEmpty() && arr[stack.peek()] < arr[i]) {
                    int idx = stack.pop();
                    result[idx] = arr[i];
                }
                stack.push(i);
            }
            return result;
        }
        
        // Largest rectangle in histogram
        public static int largestRectangle(int[] heights) {
            Stack<Integer> stack = new Stack<>();
            int maxArea = 0;
            int n = heights.length;
            
            for (int i = 0; i <= n; i++) {
                int h = (i == n) ? 0 : heights[i];
                
                while (!stack.isEmpty() && heights[stack.peek()] > h) {
                    int height = heights[stack.pop()];
                    int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                    maxArea = Math.max(maxArea, height * width);
                }
                stack.push(i);
            }
            return maxArea;
        }
    }
    
    /**
     * Pattern 17: Stack for Expression Evaluation
     * Time: O(n)
     * Space: O(n)
     */
    static class ExpressionEvaluation {
        public static int calculate(String s) {
            Stack<Integer> stack = new Stack<>();
            int num = 0;
            char sign = '+';
            
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                
                if (Character.isDigit(c)) {
                    num = num * 10 + (c - '0');
                }
                
                if ((!Character.isDigit(c) && c != ' ') || i == s.length() - 1) {
                    if (sign == '+') {
                        stack.push(num);
                    } else if (sign == '-') {
                        stack.push(-num);
                    } else if (sign == '*') {
                        stack.push(stack.pop() * num);
                    } else if (sign == '/') {
                        stack.push(stack.pop() / num);
                    }
                    sign = c;
                    num = 0;
                }
            }
            
            int result = 0;
            for (int n : stack) {
                result += n;
            }
            return result;
        }
    }
    
    /**
     * Pattern 18: Parentheses/Bracket Matching
     * Time: O(n)
     * Space: O(n)
     */
    static class ParenthesesMatching {
        // Validation
        public static boolean isValid(String s) {
            Stack<Character> stack = new Stack<>();
            Map<Character, Character> pairs = new HashMap<>();
            pairs.put(')', '(');
            pairs.put(']', '[');
            pairs.put('}', '{');
            
            for (char c : s.toCharArray()) {
                if (pairs.containsValue(c)) {
                    stack.push(c);
                } else if (pairs.containsKey(c)) {
                    if (stack.isEmpty() || stack.pop() != pairs.get(c)) {
                        return false;
                    }
                }
            }
            return stack.isEmpty();
        }
        
        // Generation
        public static List<String> generateParenthesis(int n) {
            List<String> result = new ArrayList<>();
            backtrack(result, "", 0, 0, n);
            return result;
        }
        
        private static void backtrack(List<String> result, String current, 
                                     int open, int close, int max) {
            if (current.length() == max * 2) {
                result.add(current);
                return;
            }
            
            if (open < max) {
                backtrack(result, current + "(", open + 1, close, max);
            }
            if (close < open) {
                backtrack(result, current + ")", open, close + 1, max);
            }
        }
    }
    
    // ==================== HASH-BASED PATTERNS ====================
    
    /**
     * Pattern 19: Hash Map (Frequency Counting)
     * Time: O(n)
     * Space: O(n)
     */
    static class FrequencyCounting {
        // Two Sum
        public static int[] twoSum(int[] nums, int target) {
            Map<Integer, Integer> map = new HashMap<>();
            
            for (int i = 0; i < nums.length; i++) {
                int complement = target - nums[i];
                if (map.containsKey(complement)) {
                    return new int[]{map.get(complement), i};
                }
                map.put(nums[i], i);
            }
            return new int[]{-1, -1};
        }
        
        // Group Anagrams
        public static List<List<String>> groupAnagrams(String[] strs) {
            Map<String, List<String>> map = new HashMap<>();
            
            for (String s : strs) {
                char[] chars = s.toCharArray();
                Arrays.sort(chars);
                String key = new String(chars);
                
                map.putIfAbsent(key, new ArrayList<>());
                map.get(key).add(s);
            }
            
            return new ArrayList<>(map.values());
        }
    }
    
    /**
     * Pattern 20: Hash Map (State Tracking)
     * Time: O(n)
     * Space: O(n)
     */
    static class StateTracking {
        // Subarray sum equals K
        public static int subarraySum(int[] nums, int k) {
            Map<Integer, Integer> map = new HashMap<>();
            map.put(0, 1);
            int sum = 0, count = 0;
            
            for (int num : nums) {
                sum += num;
                if (map.containsKey(sum - k)) {
                    count += map.get(sum - k);
                }
                map.put(sum, map.getOrDefault(sum, 0) + 1);
            }
            return count;
        }
        
        // Longest consecutive sequence
        public static int longestConsecutive(int[] nums) {
            Set<Integer> set = new HashSet<>();
            for (int num : nums) {
                set.add(num);
            }
            
            int maxLen = 0;
            for (int num : set) {
                if (!set.contains(num - 1)) {
                    int currentNum = num;
                    int currentLen = 1;
                    
                    while (set.contains(currentNum + 1)) {
                        currentNum++;
                        currentLen++;
                    }
                    maxLen = Math.max(maxLen, currentLen);
                }
            }
            return maxLen;
        }
    }
    
    // ==================== HEAP & PRIORITY QUEUE ====================
    
    /**
     * Pattern 21: Min/Max Heap
     * Time: O(n log k) for k elements
     * Space: O(k)
     */
    static class HeapOperations {
        // Kth largest element
        public static int findKthLargest(int[] nums, int k) {
            PriorityQueue<Integer> minHeap = new PriorityQueue<>();
            
            for (int num : nums) {
                minHeap.offer(num);
                if (minHeap.size() > k) {
                    minHeap.poll();
                }
            }
            return minHeap.peek();
        }
        
        // Top K frequent elements
        public static int[] topKFrequent(int[] nums, int k) {
            Map<Integer, Integer> freq = new HashMap<>();
            for (int num : nums) {
                freq.put(num, freq.getOrDefault(num, 0) + 1);
            }
            
            PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[1] - b[1]);
            
            for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
                heap.offer(new int[]{entry.getKey(), entry.getValue()});
                if (heap.size() > k) {
                    heap.poll();
                }
            }
            
            int[] result = new int[k];
            for (int i = 0; i < k; i++) {
                result[i] = heap.poll()[0];
            }
            return result;
        }
    }
    
    /**
     * Pattern 22: Two Heaps
     * Time: O(log n) per operation
     * Space: O(n)
     */
    static class MedianFinder {
        PriorityQueue<Integer> maxHeap; // smaller half
        PriorityQueue<Integer> minHeap; // larger half
        
        public MedianFinder() {
            maxHeap = new PriorityQueue<>((a, b) -> b - a);
            minHeap = new PriorityQueue<>();
        }
        
        public void addNum(int num) {
            maxHeap.offer(num);
            minHeap.offer(maxHeap.poll());
            
            if (minHeap.size() > maxHeap.size()) {
                maxHeap.offer(minHeap.poll());
            }
        }
        
        public double findMedian() {
            if (maxHeap.size() > minHeap.size()) {
                return maxHeap.peek();
            }
            return (maxHeap.peek() + minHeap.peek()) / 2.0;
        }
    }
    
    /**
     * Pattern 23: K-Way Merge
     * Time: O(n log k)
     * Space: O(k)
     */
    static class KWayMerge {
        static class ListNode {
            int val;
            ListNode next;
            ListNode(int val) { this.val = val; }
        }
        
        public static ListNode mergeKLists(ListNode[] lists) {
            PriorityQueue<ListNode> heap = new PriorityQueue<>((a, b) -> a.val - b.val);
            
            for (ListNode node : lists) {
                if (node != null) {
                    heap.offer(node);
                }
            }
            
            ListNode dummy = new ListNode(0);
            ListNode curr = dummy;
            
            while (!heap.isEmpty()) {
                ListNode node = heap.poll();
                curr.next = node;
                curr = curr.next;
                
                if (node.next != null) {
                    heap.offer(node.next);
                }
            }
            
            return dummy.next;
        }
        
        // Kth smallest in sorted matrix
        public static int kthSmallest(int[][] matrix, int k) {
            int n = matrix.length;
            PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
            
            for (int i = 0; i < Math.min(n, k); i++) {
                heap.offer(new int[]{matrix[i][0], i, 0});
            }
            
            int result = 0;
            while (k-- > 0) {
                int[] curr = heap.poll();
                result = curr[0];
                int row = curr[1], col = curr[2];
                
                if (col + 1 < n) {
                    heap.offer(new int[]{matrix[row][col + 1], row, col + 1});
                }
            }
            return result;
        }
    }
    
    // ==================== TREE PATTERNS ====================
    
    /**
     * Pattern 25: Tree DFS
     * Time: O(n)
     * Space: O(h) for recursion stack
     */
    static class TreeDFS {
        static class TreeNode {
            int val;
            TreeNode left, right;
            TreeNode(int val) { this.val = val; }
        }
        
        // Maximum depth
        public static int maxDepth(TreeNode root) {
            if (root == null) return 0;
            return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
        }
        
        // Path sum
        public static boolean hasPathSum(TreeNode root, int targetSum) {
            if (root == null) return false;
            if (root.left == null && root.right == null) {
                return root.val == targetSum;
            }
            return hasPathSum(root.left, targetSum - root.val) ||
                   hasPathSum(root.right, targetSum - root.val);
        }
        
        // Lowest common ancestor
        public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if (root == null || root == p || root == q) return root;
            
            TreeNode left = lowestCommonAncestor(root.left, p, q);
            TreeNode right = lowestCommonAncestor(root.right, p, q);
            
            if (left != null && right != null) return root;
            return left != null ? left : right;
        }
    }
    
    /**
     * Pattern 26: Tree BFS
     * Time: O(n)
     * Space: O(w) where w is max width
     */
    static class TreeBFS {
        static class TreeNode {
            int val;
            TreeNode left, right;
            TreeNode(int val) { this.val = val; }
        }
        
        // Level order traversal
        public static List<List<Integer>> levelOrder(TreeNode root) {
            List<List<Integer>> result = new ArrayList<>();
            if (root == null) return result;
            
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            
            while (!queue.isEmpty()) {
                int levelSize = queue.size();
                List<Integer> level = new ArrayList<>();
                
                for (int i = 0; i < levelSize; i++) {
                    TreeNode node = queue.poll();
                    level.add(node.val);
                    
                    if (node.left != null) queue.offer(node.left);
                    if (node.right != null) queue.offer(node.right);
                }
                result.add(level);
            }
            return result;
        }
        
        // Right side view
        public static List<Integer> rightSideView(TreeNode root) {
            List<Integer> result = new ArrayList<>();
            if (root == null) return result;
            
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            
            while (!queue.isEmpty()) {
                int levelSize = queue.size();
                for (int i = 0; i < levelSize; i++) {
                    TreeNode node = queue.poll();
                    if (i == levelSize - 1) {
                        result.add(node.val);
                    }
                    if (node.left != null) queue.offer(node.left);
                    if (node.right != null) queue.offer(node.right);
                }
            }
            return result;
        }
    }
    
    /**
     * Pattern 27: Binary Search Tree Properties
     * Time: O(n)
     * Space: O(h)
     */
    static class BSTOperations {
        static class TreeNode {
            int val;
            TreeNode left, right;
            TreeNode(int val) { this.val = val; }
        }
        
        // Validate BST
        public static boolean isValidBST(TreeNode root) {
            return validate(root, null, null);
        }
        
        private static boolean validate(TreeNode node, Integer min, Integer max) {
            if (node == null) return true;
            if ((min != null && node.val <= min) || (max != null && node.val >= max)) {
                return false;
            }
            return validate(node.left, min, node.val) && 
                   validate(node.right, node.val, max);
        }
        
        // Kth smallest element
        public static int kthSmallest(TreeNode root, int k) {
            int[] result = new int[]{0, 0};
            inorder(root, k, result);
            return result[1];
        }
        
        private static void inorder(TreeNode node, int k, int[] result) {
            if (node == null) return;
            inorder(node.left, k, result);
            result[0]++;
            if (result[0] == k) {
                result[1] = node.val;
                return;
            }
            inorder(node.right, k, result);
        }
    }
    
    /**
     * Pattern 28: Trie (Prefix Tree)
     * Time: O(m) per operation where m is key length
     * Space: O(ALPHABET_SIZE * n * m)
     */
    static class Trie {
        class TrieNode {
            Map<Character, TrieNode> children;
            boolean isEnd;
            
            TrieNode() {
                children = new HashMap<>();
                isEnd = false;
            }
        }
        
        private TrieNode root;
        
        public Trie() {
            root = new TrieNode();
        }
        
        public void insert(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                node.children.putIfAbsent(c, new TrieNode());
                node = node.children.get(c);
            }
            node.isEnd = true;
        }
        
        public boolean search(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                if (!node.children.containsKey(c)) {
                    return false;
                }
                node = node.children.get(c);
            }
            return node.isEnd;
        }
        
        public boolean startsWith(String prefix) {
            TrieNode node = root;
            for (char c : prefix.toCharArray()) {
                if (!node.children.containsKey(c)) {
                    return false;
                }
                node = node.children.get(c);
            }
            return true;
        }
    }
    
    // ==================== GRAPH PATTERNS ====================
    
    /**
     * Pattern 29: Graph DFS
     * Time: O(V + E)
     * Space: O(V)
     */
    static class GraphDFS {
        // Number of islands
        public static int numIslands(char[][] grid) {
            if (grid == null || grid.length == 0) return 0;
            
            int count = 0;
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length; j++) {
                    if (grid[i][j] == '1') {
                        dfs(grid, i, j);
                        count++;
                    }
                }
            }
            return count;
        }
        
        private static void dfs(char[][] grid, int i, int j) {
            if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || 
                grid[i][j] == '0') {
                return;
            }
            grid[i][j] = '0';
            dfs(grid, i + 1, j);
            dfs(grid, i - 1, j);
            dfs(grid, i, j + 1);
            dfs(grid, i, j - 1);
        }
        
        // Clone graph
        public static Node cloneGraph(Node node) {
            if (node == null) return null;
            Map<Node, Node> map = new HashMap<>();
            return clone(node, map);
        }
        
        private static Node clone(Node node, Map<Node, Node> map) {
            if (map.containsKey(node)) return map.get(node);
            
            Node copy = new Node(node.val);
            map.put(node, copy);
            
            for (Node neighbor : node.neighbors) {
                copy.neighbors.add(clone(neighbor, map));
            }
            return copy;
        }
        
        static class Node {
            int val;
            List<Node> neighbors;
            Node(int val) {
                this.val = val;
                neighbors = new ArrayList<>();
            }
        }
    }
    
    /**
     * Pattern 30: Graph BFS
     * Time: O(V + E)
     * Space: O(V)
     */
    static class GraphBFS {
        // Shortest path in binary matrix
        public static int shortestPath(int[][] grid) {
            if (grid[0][0] == 1) return -1;
            
            int n = grid.length;
            Queue<int[]> queue = new LinkedList<>();
            queue.offer(new int[]{0, 0, 1});
            grid[0][0] = 1;
            
            int[][] dirs = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
            
            while (!queue.isEmpty()) {
                int[] curr = queue.poll();
                int row = curr[0], col = curr[1], dist = curr[2];
                
                if (row == n - 1 && col == n - 1) return dist;
                
                for (int[] dir : dirs) {
                    int newRow = row + dir[0];
                    int newCol = col + dir[1];
                    
                    if (newRow >= 0 && newRow < n && newCol >= 0 && newCol < n &&
                        grid[newRow][newCol] == 0) {
                        queue.offer(new int[]{newRow, newCol, dist + 1});
                        grid[newRow][newCol] = 1;
                    }
                }
            }
            return -1;
        }
        
        // Rotting oranges
        public static int orangesRotting(int[][] grid) {
            int rows = grid.length, cols = grid[0].length;
            Queue<int[]> queue = new LinkedList<>();
            int fresh = 0;
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (grid[i][j] == 2) {
                        queue.offer(new int[]{i, j});
                    } else if (grid[i][j] == 1) {
                        fresh++;
                    }
                }
            }
            
            if (fresh == 0) return 0;
            
            int minutes = 0;
            int[][] dirs = {{0,1}, {1,0}, {0,-1}, {-1,0}};
            
            while (!queue.isEmpty()) {
                int size = queue.size();
                for (int i = 0; i < size; i++) {
                    int[] curr = queue.poll();
                    
                    for (int[] dir : dirs) {
                        int newRow = curr[0] + dir[0];
                        int newCol = curr[1] + dir[1];
                        
                        if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols &&
                            grid[newRow][newCol] == 1) {
                            grid[newRow][newCol] = 2;
                            queue.offer(new int[]{newRow, newCol});
                            fresh--;
                        }
                    }
                }
                minutes++;
            }
            
            return fresh == 0 ? minutes - 1 : -1;
        }
    }
    
    /**
     * Pattern 31: Topological Sort
     * Time: O(V + E)
     * Space: O(V)
     */
    static class TopologicalSort {
        // Course Schedule (Kahn's Algorithm)
        public static boolean canFinish(int numCourses, int[][] prerequisites) {
            List<List<Integer>> graph = new ArrayList<>();
            int[] inDegree = new int[numCourses];
            
            for (int i = 0; i < numCourses; i++) {
                graph.add(new ArrayList<>());
            }
            
            for (int[] prereq : prerequisites) {
                graph.get(prereq[1]).add(prereq[0]);
                inDegree[prereq[0]]++;
            }
            
            Queue<Integer> queue = new LinkedList<>();
            for (int i = 0; i < numCourses; i++) {
                if (inDegree[i] == 0) {
                    queue.offer(i);
                }
            }
            
            int count = 0;
            while (!queue.isEmpty()) {
                int course = queue.poll();
                count++;
                
                for (int next : graph.get(course)) {
                    inDegree[next]--;
                    if (inDegree[next] == 0) {
                        queue.offer(next);
                    }
                }
            }
            
            return count == numCourses;
        }
        
        // Find order
        public static int[] findOrder(int numCourses, int[][] prerequisites) {
            List<List<Integer>> graph = new ArrayList<>();
            int[] inDegree = new int[numCourses];
            
            for (int i = 0; i < numCourses; i++) {
                graph.add(new ArrayList<>());
            }
            
            for (int[] prereq : prerequisites) {
                graph.get(prereq[1]).add(prereq[0]);
                inDegree[prereq[0]]++;
            }
            
            Queue<Integer> queue = new LinkedList<>();
            for (int i = 0; i < numCourses; i++) {
                if (inDegree[i] == 0) {
                    queue.offer(i);
                }
            }
            
            int[] result = new int[numCourses];
            int idx = 0;
            
            while (!queue.isEmpty()) {
                int course = queue.poll();
                result[idx++] = course;
                
                for (int next : graph.get(course)) {
                    inDegree[next]--;
                    if (inDegree[next] == 0) {
                        queue.offer(next);
                    }
                }
            }
            
            return idx == numCourses ? result : new int[0];
        }
    }
    
    /**
     * Pattern 32: Union Find (Disjoint Set)
     * Time: O(α(n)) per operation
     * Space: O(n)
     */
    static class UnionFind {
        private int[] parent;
        private int[] rank;
        
        public UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                rank[i] = 0;
            }
        }
        
        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]); // Path compression
            }
            return parent[x];
        }
        
        public boolean union(int x, int y) {
            int px = find(x);
            int py = find(y);
            
            if (px == py) return false;
            
            if (rank[px] < rank[py]) {
                int temp = px;
                px = py;
                py = temp;
            }
            
            parent[py] = px;
            if (rank[px] == rank[py]) {
                rank[px]++;
            }
            return true;
        }
        
        // Number of connected components
        public static int countComponents(int n, int[][] edges) {
            UnionFind uf = new UnionFind(n);
            
            for (int[] edge : edges) {
                uf.union(edge[0], edge[1]);
            }
            
            Set<Integer> set = new HashSet<>();
            for (int i = 0; i < n; i++) {
                set.add(uf.find(i));
            }
            return set.size();
        }
    }
    
    /**
     * Pattern 33: Dijkstra's Algorithm
     * Time: O((V + E) log V)
     * Space: O(V)
     */
    static class DijkstraAlgorithm {
        public static int[] dijkstra(List<List<int[]>> graph, int start) {
            int n = graph.size();
            int[] dist = new int[n];
            Arrays.fill(dist, Integer.MAX_VALUE);
            dist[start] = 0;
            
            PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
            heap.offer(new int[]{0, start});
            
            while (!heap.isEmpty()) {
                int[] curr = heap.poll();
                int d = curr[0], node = curr[1];
                
                if (d > dist[node]) continue;
                
                for (int[] edge : graph.get(node)) {
                    int neighbor = edge[0], weight = edge[1];
                    int newDist = d + weight;
                    
                    if (newDist < dist[neighbor]) {
                        dist[neighbor] = newDist;
                        heap.offer(new int[]{newDist, neighbor});
                    }
                }
            }
            return dist;
        }
        
        // Network delay time
        public static int networkDelayTime(int[][] times, int n, int k) {
            List<List<int[]>> graph = new ArrayList<>();
            for (int i = 0; i <= n; i++) {
                graph.add(new ArrayList<>());
            }
            
            for (int[] time : times) {
                graph.get(time[0]).add(new int[]{time[1], time[2]});
            }
            
            int[] dist = dijkstra(graph, k);
            int maxDist = 0;
            
            for (int i = 1; i <= n; i++) {
                if (dist[i] == Integer.MAX_VALUE) return -1;
                maxDist = Math.max(maxDist, dist[i]);
            }
            return maxDist;
        }
    }
    
    /**
     * Pattern 34: Bellman-Ford Algorithm
     * Time: O(V * E)
     * Space: O(V)
     */
    static class BellmanFord {
        public static int[] bellmanFord(int[][] edges, int n, int start) {
            int[] dist = new int[n];
            Arrays.fill(dist, Integer.MAX_VALUE);
            dist[start] = 0;
            
            // Relax edges V-1 times
            for (int i = 0; i < n - 1; i++) {
                for (int[] edge : edges) {
                    int u = edge[0], v = edge[1], w = edge[2];
                    if (dist[u] != Integer.MAX_VALUE && dist[u] + w < dist[v]) {
                        dist[v] = dist[u] + w;
                    }
                }
            }
            
            // Check for negative cycle
            for (int[] edge : edges) {
                int u = edge[0], v = edge[1], w = edge[2];
                if (dist[u] != Integer.MAX_VALUE && dist[u] + w < dist[v]) {
                    return null; // Negative cycle exists
                }
            }
            return dist;
        }
    }
    
    /**
     * Pattern 35: Floyd-Warshall Algorithm
     * Time: O(V³)
     * Space: O(V²)
     */
    static class FloydWarshall {
        public static int[][] floydWarshall(int n, int[][] edges) {
            int[][] dist = new int[n][n];
            
            for (int i = 0; i < n; i++) {
                Arrays.fill(dist[i], Integer.MAX_VALUE / 2);
                dist[i][i] = 0;
            }
            
            for (int[] edge : edges) {
                dist[edge[0]][edge[1]] = edge[2];
            }
            
            for (int k = 0; k < n; k++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
            return dist;
        }
    }
    
    /**
     * Pattern 36: Minimum Spanning Tree (Kruskal's)
     * Time: O(E log E)
     * Space: O(V)
     */
    static class KruskalMST {
        public static int kruskal(int n, int[][] edges) {
            Arrays.sort(edges, (a, b) -> a[2] - b[2]);
            UnionFind uf = new UnionFind(n);
            
            int mstWeight = 0;
            int mstEdges = 0;
            
            for (int[] edge : edges) {
                if (uf.union(edge[0], edge[1])) {
                    mstWeight += edge[2];
                    mstEdges++;
                    if (mstEdges == n - 1) break;
                }
            }
            return mstWeight;
        }
    }
    
    /**
     * Pattern 36: Minimum Spanning Tree (Prim's)
     * Time: O(E log V)
     * Space: O(V)
     */
    static class PrimMST {
        public static int prim(int n, List<List<int[]>> graph) {
            boolean[] visited = new boolean[n];
            PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
            
            visited[0] = true;
            for (int[] edge : graph.get(0)) {
                heap.offer(new int[]{edge[1], edge[0]});
            }
            
            int mstWeight = 0;
            int edges = 0;
            
            while (!heap.isEmpty() && edges < n - 1) {
                int[] curr = heap.poll();
                int weight = curr[0], node = curr[1];
                
                if (visited[node]) continue;
                
                visited[node] = true;
                mstWeight += weight;
                edges++;
                
                for (int[] edge : graph.get(node)) {
                    if (!visited[edge[0]]) {
                        heap.offer(new int[]{edge[1], edge[0]});
                    }
                }
            }
            return mstWeight;
        }
    }
    
    // ==================== BINARY SEARCH VARIATIONS ====================
    
    /**
     * Pattern 37: Classic Binary Search
     * Time: O(log n)
     * Space: O(1)
     */
    static class BinarySearch {
        public static int binarySearch(int[] arr, int target) {
            int left = 0, right = arr.length - 1;
            
            while (left <= right) {
                int mid = left + (right - left) / 2;
                
                if (arr[mid] == target) {
                    return mid;
                } else if (arr[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return -1;
        }
        
        // Find first occurrence
        public static int findFirst(int[] arr, int target) {
            int left = 0, right = arr.length - 1;
            int result = -1;
            
            while (left <= right) {
                int mid = left + (right - left) / 2;
                
                if (arr[mid] == target) {
                    result = mid;
                    right = mid - 1;
                } else if (arr[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return result;
        }
    }
    
    /**
     * Pattern 38: Modified Binary Search
     * Time: O(log n)
     * Space: O(1)
     */
    static class ModifiedBinarySearch {
        // Search in rotated sorted array
        public static int searchRotated(int[] arr, int target) {
            int left = 0, right = arr.length - 1;
            
            while (left <= right) {
                int mid = left + (right - left) / 2;
                
                if (arr[mid] == target) return mid;
                
                // Left half is sorted
                if (arr[left] <= arr[mid]) {
                    if (arr[left] <= target && target < arr[mid]) {
                        right = mid - 1;
                    } else {
                        left = mid + 1;
                    }
                }
                // Right half is sorted
                else {
                    if (arr[mid] < target && target <= arr[right]) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
            }
            return -1;
        }
        
        // Find minimum in rotated sorted array
        public static int findMin(int[] arr) {
            int left = 0, right = arr.length - 1;
            
            while (left < right) {
                int mid = left + (right - left) / 2;
                
                if (arr[mid] > arr[right]) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            return arr[left];
        }
    }
    
    /**
     * Pattern 39: Binary Search on Answer
     * Time: O(n log(max-min))
     * Space: O(1)
     */
    static class BinarySearchOnAnswer {
        // Koko eating bananas
        public static int minEatingSpeed(int[] piles, int h) {
            int left = 1, right = Arrays.stream(piles).max().getAsInt();
            
            while (left < right) {
                int mid = left + (right - left) / 2;
                
                if (canEatAll(piles, mid, h)) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            return left;
        }
        
        private static boolean canEatAll(int[] piles, int k, int h) {
            int hours = 0;
            for (int pile : piles) {
                hours += (pile + k - 1) / k;
                if (hours > h) return false;
            }
            return true;
        }
        
        // Split array largest sum
        public static int splitArray(int[] nums, int m) {
            int left = Arrays.stream(nums).max().getAsInt();
            int right = Arrays.stream(nums).sum();
            
            while (left < right) {
                int mid = left + (right - left) / 2;
                
                if (canSplit(nums, m, mid)) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            return left;
        }
        
        private static boolean canSplit(int[] nums, int m, int maxSum) {
            int subarrays = 1;
            int currentSum = 0;
            
            for (int num : nums) {
                if (currentSum + num > maxSum) {
                    subarrays++;
                    currentSum = num;
                    if (subarrays > m) return false;
                } else {
                    currentSum += num;
                }
            }
            return true;
        }
    }
    
    // ==================== DYNAMIC PROGRAMMING ====================
    
    /**
     * Pattern 40: 1D DP (Linear)
     * Time: O(n)
     * Space: O(n) or O(1) optimized
     */
    static class LinearDP {
        // Climbing stairs
        public static int climbStairs(int n) {
            if (n <= 2) return n;
            
            int prev2 = 1, prev1 = 2;
            for (int i = 3; i <= n; i++) {
                int curr = prev1 + prev2;
                prev2 = prev1;
                prev1 = curr;
            }
            return prev1;
        }
        
        // House robber
        public static int rob(int[] nums) {
            if (nums.length == 0) return 0;
            if (nums.length == 1) return nums[0];
            
            int prev2 = 0, prev1 = nums[0];
            for (int i = 1; i < nums.length; i++) {
                int curr = Math.max(prev1, prev2 + nums[i]);
                prev2 = prev1;
                prev1 = curr;
            }
            return prev1;
        }
        
        // Decode ways
        public static int numDecodings(String s) {
            if (s.charAt(0) == '0') return 0;
            
            int n = s.length();
            int prev2 = 1, prev1 = 1;
            
            for (int i = 1; i < n; i++) {
                int curr = 0;
                
                if (s.charAt(i) != '0') {
                    curr = prev1;
                }
                
                int twoDigit = Integer.parseInt(s.substring(i-1, i+1));
                if (twoDigit >= 10 && twoDigit <= 26) {
                    curr += prev2;
                }
                
                prev2 = prev1;
                prev1 = curr;
            }
            return prev1;
        }
    }
    
    /**
     * Pattern 41: 2D DP (Grid/Matrix)
     * Time: O(m * n)
     * Space: O(m * n) or O(n) optimized
     */
    static class GridDP {
        // Unique paths
        public static int uniquePaths(int m, int n) {
            int[][] dp = new int[m][n];
            
            for (int i = 0; i < m; i++) {
                dp[i][0] = 1;
            }
            for (int j = 0; j < n; j++) {
                dp[0][j] = 1;
            }
            
            for (int i = 1; i < m; i++) {
                for (int j = 1; j < n; j++) {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
            return dp[m-1][n-1];
        }
        
        // Longest common subsequence
        public static int longestCommonSubsequence(String text1, String text2) {
            int m = text1.length(), n = text2.length();
            int[][] dp = new int[m + 1][n + 1];
            
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= n; j++) {
                    if (text1.charAt(i-1) == text2.charAt(j-1)) {
                        dp[i][j] = dp[i-1][j-1] + 1;
                    } else {
                        dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                    }
                }
            }
            return dp[m][n];
        }
        
        // Edit distance
        public static int minDistance(String word1, String word2) {
            int m = word1.length(), n = word2.length();
            int[][] dp = new int[m + 1][n + 1];
            
            for (int i = 0; i <= m; i++) {
                dp[i][0] = i;
            }
            for (int j = 0; j <= n; j++) {
                dp[0][j] = j;
            }
            
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= n; j++) {
                    if (word1.charAt(i-1) == word2.charAt(j-1)) {
                        dp[i][j] = dp[i-1][j-1];
                    } else {
                        dp[i][j] = 1 + Math.min(dp[i-1][j],    // delete
                                        Math.min(dp[i][j-1],    // insert
                                                dp[i-1][j-1])); // replace
                    }
                }
            }
            return dp[m][n];
        }
    }
    
    /**
     * Pattern 42: Knapsack Pattern (0/1)
     * Time: O(n * capacity)
     * Space: O(n * capacity)
     */
    static class KnapsackDP {
        public static int knapsack(int[] weights, int[] values, int capacity) {
            int n = weights.length;
            int[][] dp = new int[n + 1][capacity + 1];
            
            for (int i = 1; i <= n; i++) {
                for (int w = 0; w <= capacity; w++) {
                    if (weights[i-1] <= w) {
                        dp[i][w] = Math.max(dp[i-1][w],
                                          dp[i-1][w - weights[i-1]] + values[i-1]);
                    } else {
                        dp[i][w] = dp[i-1][w];
                    }
                }
            }
            return dp[n][capacity];
        }
        
        // Partition equal subset sum
        public static boolean canPartition(int[] nums) {
            int sum = Arrays.stream(nums).sum();
            if (sum % 2 != 0) return false;
            
            int target = sum / 2;
            boolean[] dp = new boolean[target + 1];
            dp[0] = true;
            
            for (int num : nums) {
                for (int j = target; j >= num; j--) {
                    dp[j] = dp[j] || dp[j - num];
                }
            }
            return dp[target];
        }
    }
    
    /**
     * Pattern 43: Unbounded Knapsack
     * Time: O(n * target)
     * Space: O(target)
     */
    static class UnboundedKnapsack {
        // Coin change
        public static int coinChange(int[] coins, int amount) {
            int[] dp = new int[amount + 1];
            Arrays.fill(dp, amount + 1);
            dp[0] = 0;
            
            for (int i = 1; i <= amount; i++) {
                for (int coin : coins) {
                    if (i >= coin) {
                        dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                    }
                }
            }
            return dp[amount] > amount ? -1 : dp[amount];
        }
        
        // Coin change 2 (number of ways)
        public static int change(int amount, int[] coins) {
            int[] dp = new int[amount + 1];
            dp[0] = 1;
            
            for (int coin : coins) {
                for (int i = coin; i <= amount; i++) {
                    dp[i] += dp[i - coin];
                }
            }
            return dp[amount];
        }
    }
    
    /**
     * Pattern 44: Longest Increasing Subsequence
     * Time: O(n²) or O(n log n) with binary search
     * Space: O(n)
     */
    static class LIS {
        // O(n²) approach
        public static int lengthOfLIS(int[] nums) {
            int n = nums.length;
            int[] dp = new int[n];
            Arrays.fill(dp, 1);
            
            for (int i = 1; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    if (nums[j] < nums[i]) {
                        dp[i] = Math.max(dp[i], dp[j] + 1);
                    }
                }
            }
            return Arrays.stream(dp).max().getAsInt();
        }
        
        // O(n log n) approach with binary search
        public static int lengthOfLISOptimized(int[] nums) {
            List<Integer> tails = new ArrayList<>();
            
            for (int num : nums) {
                int pos = Collections.binarySearch(tails, num);
                if (pos < 0) pos = -(pos + 1);
                
                if (pos == tails.size()) {
                    tails.add(num);
                } else {
                    tails.set(pos, num);
                }
            }
            return tails.size();
        }
    }
    
    /**
     * Pattern 45: Palindrome DP
     * Time: O(n²)
     * Space: O(n²)
     */
    static class PalindromeDP {
        // Longest palindromic substring
        public static String longestPalindrome(String s) {
            int n = s.length();
            boolean[][] dp = new boolean[n][n];
            int start = 0, maxLen = 1;
            
            for (int i = 0; i < n; i++) {
                dp[i][i] = true;
            }
            
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i < n - len + 1; i++) {
                    int j = i + len - 1;
                    
                    if (s.charAt(i) == s.charAt(j)) {
                        dp[i][j] = (len == 2) || dp[i+1][j-1];
                        if (dp[i][j] && len > maxLen) {
                            start = i;
                            maxLen = len;
                        }
                    }
                }
            }
            return s.substring(start, start + maxLen);
        }
        
        // Longest palindromic subsequence
        public static int longestPalindromeSubseq(String s) {
            int n = s.length();
            int[][] dp = new int[n][n];
            
            for (int i = 0; i < n; i++) {
                dp[i][i] = 1;
            }
            
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i < n - len + 1; i++) {
                    int j = i + len - 1;
                    
                    if (s.charAt(i) == s.charAt(j)) {
                        dp[i][j] = dp[i+1][j-1] + 2;
                    } else {
                        dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
                    }
                }
            }
            return dp[0][n-1];
        }
    }
    
    /**
     * Pattern 47: DP with Bitmask
     * Time: O(2^n * n²)
     * Space: O(2^n * n)
     */
    static class BitmaskDP {
        // Traveling Salesman Problem
        public static int tsp(int[][] graph) {
            int n = graph.length;
            int[][] dp = new int[1 << n][n];
            
            for (int[] row : dp) {
                Arrays.fill(row, Integer.MAX_VALUE / 2);
            }
            dp[1][0] = 0;
            
            for (int mask = 1; mask < (1 << n); mask++) {
                for (int u = 0; u < n; u++) {
                    if ((mask & (1 << u)) != 0) {
                        for (int v = 0; v < n; v++) {
                            if ((mask & (1 << v)) == 0) {
                                int newMask = mask | (1 << v);
                                dp[newMask][v] = Math.min(dp[newMask][v],
                                                         dp[mask][u] + graph[u][v]);
                            }
                        }
                    }
                }
            }
            
            int result = Integer.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                result = Math.min(result, dp[(1 << n) - 1][i] + graph[i][0]);
            }
            return result;
        }
    }
    
    /**
     * Pattern 48: State Machine DP
     * Time: O(n)
     * Space: O(1)
     */
    static class StateMachineDP {
        // Best time to buy and sell stock with cooldown
        public static int maxProfit(int[] prices) {
            int hold = -prices[0];
            int sold = 0;
            int rest = 0;
            
            for (int i = 1; i < prices.length; i++) {
                int prevHold = hold;
                int prevSold = sold;
                int prevRest = rest;
                
                hold = Math.max(prevHold, prevRest - prices[i]);
                sold = prevHold + prices[i];
                rest = Math.max(prevRest, prevSold);
            }
            return Math.max(sold, rest);
        }
    }
    
    /**
     * Pattern 49: Interval DP
     * Time: O(n³)
     * Space: O(n²)
     */
    static class IntervalDP {
        // Burst balloons
        public static int maxCoins(int[] nums) {
            int n = nums.length;
            int[] arr = new int[n + 2];
            arr[0] = arr[n + 1] = 1;
            for (int i = 0; i < n; i++) {
                arr[i + 1] = nums[i];
            }
            
            int[][] dp = new int[n + 2][n + 2];
            
            for (int len = 1; len <= n; len++) {
                for (int i = 1; i <= n - len + 1; i++) {
                    int j = i + len - 1;
                    
                    for (int k = i; k <= j; k++) {
                        dp[i][j] = Math.max(dp[i][j],
                                          dp[i][k-1] + dp[k+1][j] +
                                          arr[i-1] * arr[k] * arr[j+1]);
                    }
                }
            }
            return dp[1][n];
        }
    }
    
    /**
     * Pattern 50: Matrix Chain Multiplication
     * Time: O(n³)
     * Space: O(n²)
     */
    static class MatrixChainMultiplication {
        public static int matrixChainOrder(int[] dims) {
            int n = dims.length - 1;
            int[][] dp = new int[n][n];
            
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i < n - len + 1; i++) {
                    int j = i + len - 1;
                    dp[i][j] = Integer.MAX_VALUE;
                    
                    for (int k = i; k < j; k++) {
                        int cost = dp[i][k] + dp[k+1][j] +
                                  dims[i] * dims[k+1] * dims[j+1];
                        dp[i][j] = Math.min(dp[i][j], cost);
                    }
                }
            }
            return dp[0][n-1];
        }
    }
    
    // ==================== GREEDY ALGORITHMS ====================
    
    /**
     * Pattern 51: Greedy (Activity Selection)
     * Time: O(n log n)
     * Space: O(1)
     */
    static class ActivitySelection {
        // Non-overlapping intervals
        public static int eraseOverlapIntervals(int[][] intervals) {
            if (intervals.length == 0) return 0;
            
            Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
            int count = 0;
            int end = Integer.MIN_VALUE;
            
            for (int[] interval : intervals) {
                if (interval[0] >= end) {
                    count++;
                    end = interval[1];
                }
            }
            return intervals.length - count;
        }
        
        // Meeting rooms II
        public static int minMeetingRooms(int[][] intervals) {
            int[] start = new int[intervals.length];
            int[] end = new int[intervals.length];
            
            for (int i = 0; i < intervals.length; i++) {
                start[i] = intervals[i][0];
                end[i] = intervals[i][1];
            }
            
            Arrays.sort(start);
            Arrays.sort(end);
            
            int rooms = 0, endIdx = 0;
            for (int i = 0; i < start.length; i++) {
                if (start[i] < end[endIdx]) {
                    rooms++;
                } else {
                    endIdx++;
                }
            }
            return rooms;
        }
    }
    
    /**
     * Pattern 52: Greedy (Two Pointers/Sorting)
     * Time: O(n log n)
     * Space: O(1)
     */
    static class GreedyTwoPointers {
        // Assign cookies
        public static int findContentChildren(int[] g, int[] s) {
            Arrays.sort(g);
            Arrays.sort(s);
            
            int i = 0, j = 0;
            while (i < g.length && j < s.length) {
                if (s[j] >= g[i]) {
                    i++;
                }
                j++;
            }
            return i;
        }
        
        // Boats to save people
        public static int numRescueBoats(int[] people, int limit) {
            Arrays.sort(people);
            int left = 0, right = people.length - 1;
            int boats = 0;
            
            while (left <= right) {
                if (people[left] + people[right] <= limit) {
                    left++;
                }
                right--;
                boats++;
            }
            return boats;
        }
    }
    
    /**
     * Pattern 53: Greedy (Priority Queue)
     * Time: O(n log n)
     * Space: O(n)
     */
    static class GreedyPriorityQueue {
        // Task scheduler
        public static int leastInterval(char[] tasks, int n) {
            int[] freq = new int[26];
            for (char task : tasks) {
                freq[task - 'A']++;
            }
            
            PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
            for (int f : freq) {
                if (f > 0) maxHeap.offer(f);
            }
            
            int time = 0;
            while (!maxHeap.isEmpty()) {
                List<Integer> temp = new ArrayList<>();
                
                for (int i = 0; i <= n; i++) {
                    if (!maxHeap.isEmpty()) {
                        int f = maxHeap.poll();
                        if (f > 1) temp.add(f - 1);
                    }
                    time++;
                    if (maxHeap.isEmpty() && temp.isEmpty()) break;
                }
                
                for (int f : temp) {
                    maxHeap.offer(f);
                }
            }
            return time;
        }
    }
    
    // ==================== BACKTRACKING ====================
    
    /**
     * Pattern 54: Backtracking (Combinations)
     * Time: O(2^n)
     * Space: O(n)
     */
    static class BacktrackingCombinations {
        // Subsets
        public static List<List<Integer>> subsets(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            backtrack(nums, 0, new ArrayList<>(), result);
            return result;
        }
        
        private static void backtrack(int[] nums, int start, 
                                     List<Integer> path, List<List<Integer>> result) {
            result.add(new ArrayList<>(path));
            
            for (int i = start; i < nums.length; i++) {
                path.add(nums[i]);
                backtrack(nums, i + 1, path, result);
                path.remove(path.size() - 1);
            }
        }
        
        // Combination sum
        public static List<List<Integer>> combinationSum(int[] candidates, int target) {
            List<List<Integer>> result = new ArrayList<>();
            Arrays.sort(candidates);
            backtrackSum(candidates, target, 0, new ArrayList<>(), result);
            return result;
        }
        
        private static void backtrackSum(int[] candidates, int target, int start,
                                        List<Integer> path, List<List<Integer>> result) {
            if (target == 0) {
                result.add(new ArrayList<>(path));
                return;
            }
            
            for (int i = start; i < candidates.length; i++) {
                if (candidates[i] > target) break;
                path.add(candidates[i]);
                backtrackSum(candidates, target - candidates[i], i, path, result);
                path.remove(path.size() - 1);
            }
        }
    }
    
    /**
     * Pattern 55: Backtracking (Permutations)
     * Time: O(n!)
     * Space: O(n)
     */
    static class BacktrackingPermutations {
        // Permutations
        public static List<List<Integer>> permute(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            backtrack(nums, new ArrayList<>(), new boolean[nums.length], result);
            return result;
        }
        
        private static void backtrack(int[] nums, List<Integer> path, 
                                     boolean[] used, List<List<Integer>> result) {
            if (path.size() == nums.length) {
                result.add(new ArrayList<>(path));
                return;
            }
            
            for (int i = 0; i < nums.length; i++) {
                if (used[i]) continue;
                
                path.add(nums[i]);
                used[i] = true;
                backtrack(nums, path, used, result);
                path.remove(path.size() - 1);
                used[i] = false;
            }
        }
    }
    
    /**
     * Pattern 56: Backtracking (Board/Grid)
     * Time: O(m * n * 4^L) where L is word length
     * Space: O(L)
     */
    static class BacktrackingBoard {
        // Word search
        public static boolean exist(char[][] board, String word) {
            int m = board.length, n = board[0].length;
            
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (backtrack(board, word, i, j, 0)) {
                        return true;
                    }
                }
            }
            return false;
        }
        
        private static boolean backtrack(char[][] board, String word, 
                                        int row, int col, int index) {
            if (index == word.length()) return true;
            
            if (row < 0 || row >= board.length || col < 0 || col >= board[0].length ||
                board[row][col] != word.charAt(index)) {
                return false;
            }
            
            char temp = board[row][col];
            board[row][col] = '#';
            
            boolean found = backtrack(board, word, row + 1, col, index + 1) ||
                           backtrack(board, word, row - 1, col, index + 1) ||
                           backtrack(board, word, row, col + 1, index + 1) ||
                           backtrack(board, word, row, col - 1, index + 1);
            
            board[row][col] = temp;
            return found;
        }
        
        // N-Queens
        public static List<List<String>> solveNQueens(int n) {
            List<List<String>> result = new ArrayList<>();
            char[][] board = new char[n][n];
            for (int i = 0; i < n; i++) {
                Arrays.fill(board[i], '.');
            }
            backtrackQueens(board, 0, result);
            return result;
        }
        
        private static void backtrackQueens(char[][] board, int row, 
                                           List<List<String>> result) {
            if (row == board.length) {
                result.add(construct(board));
                return;
            }
            
            for (int col = 0; col < board.length; col++) {
                if (isValid(board, row, col)) {
                    board[row][col] = 'Q';
                    backtrackQueens(board, row + 1, result);
                    board[row][col] = '.';
                }
            }
        }
        
        private static boolean isValid(char[][] board, int row, int col) {
            int n = board.length;
            
            for (int i = 0; i < row; i++) {
                if (board[i][col] == 'Q') return false;
            }
            
            for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
                if (board[i][j] == 'Q') return false;
            }
            
            for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
                if (board[i][j] == 'Q') return false;
            }
            
            return true;
        }
        
        private static List<String> construct(char[][] board) {
            List<String> result = new ArrayList<>();
            for (char[] row : board) {
                result.add(new String(row));
            }
            return result;
        }
    }
    
    // ==================== BIT MANIPULATION ====================
    
    /**
     * Pattern 57: Basic Bit Operations
     * Time: O(1) or O(log n)
     * Space: O(1)
     */
    static class BitOperations {
        // Check if bit at position i is set
        public static boolean isBitSet(int num, int i) {
            return ((num >> i) & 1) == 1;
        }
        
        // Set bit at position i
        public static int setBit(int num, int i) {
            return num | (1 << i);
        }
        
        // Clear bit at position i
        public static int clearBit(int num, int i) {
            return num & ~(1 << i);
        }
        
        // Toggle bit at position i
        public static int toggleBit(int num, int i) {
            return num ^ (1 << i);
        }
        
        // Clear rightmost set bit
        public static int clearRightmostBit(int num) {
            return num & (num - 1);
        }
        
        // Get rightmost set bit
        public static int getRightmostBit(int num) {
            return num & -num;
        }
        
        // Count set bits
        public static int countSetBits(int num) {
            int count = 0;
            while (num != 0) {
                num &= (num - 1);
                count++;
            }
            return count;
        }
        
        // Power of two
        public static boolean isPowerOfTwo(int n) {
            return n > 0 && (n & (n - 1)) == 0;
        }
    }
    
    /**
     * Pattern 58: XOR Tricks
     * Time: O(n)
     * Space: O(1)
     */
    static class XORTricks {
        // Find single number (all others appear twice)
        public static int singleNumber(int[] nums) {
            int result = 0;
            for (int num : nums) {
                result ^= num;
            }
            return result;
        }
        
        // Find two unique numbers (all others appear twice)
        public static int[] singleNumberIII(int[] nums) {
            int xor = 0;
            for (int num : nums) {
                xor ^= num;
            }
            
            int rightmostBit = xor & -xor;
            int num1 = 0, num2 = 0;
            
            for (int num : nums) {
                if ((num & rightmostBit) != 0) {
                    num1 ^= num;
                } else {
                    num2 ^= num;
                }
            }
            return new int[]{num1, num2};
        }
        
        // Missing number
        public static int missingNumber(int[] nums) {
            int result = nums.length;
            for (int i = 0; i < nums.length; i++) {
                result ^= i ^ nums[i];
            }
            return result;
        }
    }
    
    /**
     * Pattern 59: Bit Masking
     * Time: O(2^n) for subset iteration
     * Space: O(1)
     */
    static class BitMasking {
        // Generate all subsets
        public static List<List<Integer>> generateSubsets(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            int n = nums.length;
            
            for (int mask = 0; mask < (1 << n); mask++) {
                List<Integer> subset = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if ((mask & (1 << i)) != 0) {
                        subset.add(nums[i]);
                    }
                }
                result.add(subset);
            }
            return result;
        }
        
        // Maximum product of word lengths
        public static int maxProduct(String[] words) {
            int n = words.length;
            int[] masks = new int[n];
            
            for (int i = 0; i < n; i++) {
                for (char c : words[i].toCharArray()) {
                    masks[i] |= (1 << (c - 'a'));
                }
            }
            
            int maxProd = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if ((masks[i] & masks[j]) == 0) {
                        maxProd = Math.max(maxProd, words[i].length() * words[j].length());
                    }
                }
            }
            return maxProd;
        }
    }
    
    // ==================== MATHEMATICAL & GEOMETRIC ====================
    
    /**
     * Pattern 60: Math (Prime Numbers)
     * Time: O(n log log n) for sieve
     * Space: O(n)
     */
    static class PrimeNumbers {
        // Sieve of Eratosthenes
        public static List<Integer> sieveOfEratosthenes(int n) {
            boolean[] isPrime = new boolean[n + 1];
            Arrays.fill(isPrime, true);
            isPrime[0] = isPrime[1] = false;
            
            for (int i = 2; i * i <= n; i++) {
                if (isPrime[i]) {
                    for (int j = i * i; j <= n; j += i) {
                        isPrime[j] = false;
                    }
                }
            }
            
            List<Integer> primes = new ArrayList<>();
            for (int i = 2; i <= n; i++) {
                if (isPrime[i]) primes.add(i);
            }
            return primes;
        }
        
        // Check if prime
        public static boolean isPrime(int n) {
            if (n < 2) return false;
            for (int i = 2; i * i <= n; i++) {
                if (n % i == 0) return false;
            }
            return true;
        }
        
        // Count primes
        public static int countPrimes(int n) {
            if (n <= 2) return 0;
            
            boolean[] isPrime = new boolean[n];
            Arrays.fill(isPrime, true);
            
            for (int i = 2; i * i < n; i++) {
                if (isPrime[i]) {
                    for (int j = i * i; j < n; j += i) {
                        isPrime[j] = false;
                    }
                }
            }
            
            int count = 0;
            for (int i = 2; i < n; i++) {
                if (isPrime[i]) count++;
            }
            return count;
        }
    }
    
    /**
     * Pattern 61: Math (GCD/LCM)
     * Time: O(log min(a, b))
     * Space: O(1)
     */
    static class GCDLCM {
        public static int gcd(int a, int b) {
            while (b != 0) {
                int temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }
        
        public static int lcm(int a, int b) {
            return (a * b) / gcd(a, b);
        }
        
        // GCD of array
        public static int gcdArray(int[] nums) {
            int result = nums[0];
            for (int i = 1; i < nums.length; i++) {
                result = gcd(result, nums[i]);
            }
            return result;
        }
    }
    
    /**
     * Pattern 62: Math (Combinatorics)
     * Time: O(k)
     * Space: O(1)
     */
    static class Combinatorics {
        // Combinations C(n, k)
        public static long combinations(int n, int k) {
            if (k > n - k) k = n - k;
            
            long result = 1;
            for (int i = 0; i < k; i++) {
                result = result * (n - i) / (i + 1);
            }
            return result;
        }
        
        // Permutations P(n, k)
        public static long permutations(int n, int k) {
            long result = 1;
            for (int i = 0; i < k; i++) {
                result *= (n - i);
            }
            return result;
        }
        
        // Pascal's triangle
        public static List<List<Integer>> generate(int numRows) {
            List<List<Integer>> result = new ArrayList<>();
            
            for (int i = 0; i < numRows; i++) {
                List<Integer> row = new ArrayList<>();
                for (int j = 0; j <= i; j++) {
                    if (j == 0 || j == i) {
                        row.add(1);
                    } else {
                        row.add(result.get(i-1).get(j-1) + result.get(i-1).get(j));
                    }
                }
                result.add(row);
            }
            return result;
        }
    }
    
    /**
     * Pattern 63: Catalan Numbers
     * Time: O(n²)
     * Space: O(n)
     */
    static class CatalanNumbers {
        public static int catalan(int n) {
            if (n <= 1) return 1;
            
            int[] dp = new int[n + 1];
            dp[0] = dp[1] = 1;
            
            for (int i = 2; i <= n; i++) {
                for (int j = 0; j < i; j++) {
                    dp[i] += dp[j] * dp[i-1-j];
                }
            }
            return dp[n];
        }
        
        // Number of unique BSTs
        public static int numTrees(int n) {
            return catalan(n);
        }
    }
    
    /**
     * Pattern 64: Geometry (Line Sweep)
     * Time: O(n log n)
     * Space: O(n)
     */
    static class LineSweep {
        // Meeting rooms II
        public static int minMeetingRooms(int[][] intervals) {
            List<int[]> events = new ArrayList<>();
            
            for (int[] interval : intervals) {
                events.add(new int[]{interval[0], 1});
                events.add(new int[]{interval[1], -1});
            }
            
            events.sort((a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
            
            int active = 0, maxActive = 0;
            for (int[] event : events) {
                active += event[1];
                maxActive = Math.max(maxActive, active);
            }
            return maxActive;
        }
    }
    
    /**
     * Pattern 65: Geometry (Manhattan Distance)
     * Time: O(n log n)
     * Space: O(n)
     */
    static class ManhattanDistance {
        // Best meeting point
        public static int minTotalDistance(int[][] grid) {
            List<Integer> rows = new ArrayList<>();
            List<Integer> cols = new ArrayList<>();
            
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length; j++) {
                    if (grid[i][j] == 1) {
                        rows.add(i);
                        cols.add(j);
                    }
                }
            }
            
            Collections.sort(cols);
            int medianRow = rows.get(rows.size() / 2);
            int medianCol = cols.get(cols.size() / 2);
            
            int distance = 0;
            for (int row : rows) {
                distance += Math.abs(row - medianRow);
            }
            for (int col : cols) {
                distance += Math.abs(col - medianCol);
            }
            return distance;
        }
    }
    
    /**
     * Pattern 66: Geometry (Euclidean Distance)
     * Time: O(n log k)
     * Space: O(k)
     */
    static class EuclideanDistance {
        // K closest points to origin
        public static int[][] kClosest(int[][] points, int k) {
            PriorityQueue<int[]> maxHeap = new PriorityQueue<>((a, b) -> 
                (b[0]*b[0] + b[1]*b[1]) - (a[0]*a[0] + a[1]*a[1])
            );
            
            for (int[] point : points) {
                maxHeap.offer(point);
                if (maxHeap.size() > k) {
                    maxHeap.poll();
                }
            }
            
            int[][] result = new int[k][2];
            for (int i = 0; i < k; i++) {
                result[i] = maxHeap.poll();
            }
            return result;
        }
        
        public static double distance(int[] p1, int[] p2) {
            return Math.sqrt(Math.pow(p1[0] - p2[0], 2) + Math.pow(p1[1] - p2[1], 2));
        }
        
        public static int squaredDistance(int[] p1, int[] p2) {
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]);
        }
    }
    
    // ==================== ADVANCED DATA STRUCTURES ====================
    
    /**
     * Pattern 67: Binary Indexed Tree (Fenwick Tree)
     * Time: O(log n) per operation
     * Space: O(n)
     */
    static class BinaryIndexedTree {
        private int[] tree;
        private int n;
        
        public BinaryIndexedTree(int n) {
            this.n = n;
            this.tree = new int[n + 1];
        }
        
        public void update(int i, int delta) {
            while (i <= n) {
                tree[i] += delta;
                i += i & (-i);
            }
        }
        
        public int query(int i) {
            int sum = 0;
            while (i > 0) {
                sum += tree[i];
                i -= i & (-i);
            }
            return sum;
        }
        
        public int rangeQuery(int l, int r) {
            return query(r) - query(l - 1);
        }
    }
    
    /**
     * Pattern 68: Segment Tree
     * Time: O(log n) per operation
     * Space: O(n)
     */
    static class SegmentTree {
        private int[] tree;
        private int n;
        
        public SegmentTree(int[] arr) {
            n = arr.length;
            tree = new int[4 * n];
            build(arr, 0, 0, n - 1);
        }
        
        private void build(int[] arr, int node, int start, int end) {
            if (start == end) {
                tree[node] = arr[start];
            } else {
                int mid = (start + end) / 2;
                build(arr, 2*node+1, start, mid);
                build(arr, 2*node+2, mid+1, end);
                tree[node] = tree[2*node+1] + tree[2*node+2];
            }
        }
        
        public void update(int idx, int val) {
            update(0, 0, n - 1, idx, val);
        }
        
        private void update(int node, int start, int end, int idx, int val) {
            if (start == end) {
                tree[node] = val;
            } else {
                int mid = (start + end) / 2;
                if (idx <= mid) {
                    update(2*node+1, start, mid, idx, val);
                } else {
                    update(2*node+2, mid+1, end, idx, val);
                }
                tree[node] = tree[2*node+1] + tree[2*node+2];
            }
        }
        
        public int query(int l, int r) {
            return query(0, 0, n - 1, l, r);
        }
        
        private int query(int node, int start, int end, int l, int r) {
            if (r < start || end < l) return 0;
            if (l <= start && end <= r) return tree[node];
            
            int mid = (start + end) / 2;
            return query(2*node+1, start, mid, l, r) + 
                   query(2*node+2, mid+1, end, l, r);
        }
    }
    
    /**
     * Pattern 69: Sparse Table
     * Time: O(1) query, O(n log n) preprocessing
     * Space: O(n log n)
     */
    static class SparseTable {
        private int[][] st;
        private int n;
        
        public SparseTable(int[] arr) {
            n = arr.length;
            int k = (int)(Math.log(n) / Math.log(2)) + 1;
            st = new int[n][k];
            
            for (int i = 0; i < n; i++) {
                st[i][0] = arr[i];
            }
            
            for (int j = 1; (1 << j) <= n; j++) {
                for (int i = 0; i + (1 << j) - 1 < n; i++) {
                    st[i][j] = Math.min(st[i][j-1], st[i + (1 << (j-1))][j-1]);
                }
            }
        }
        
        public int query(int l, int r) {
            int j = (int)(Math.log(r - l + 1) / Math.log(2));
            return Math.min(st[l][j], st[r - (1 << j) + 1][j]);
        }
    }
    
    /**
     * Pattern 70: Sqrt Decomposition
     * Time: O(sqrt(n)) per operation
     * Space: O(sqrt(n))
     */
    static class SqrtDecomposition {
        private int[] arr;
        private int[] blocks;
        private int blockSize;
        private int numBlocks;
        
        public SqrtDecomposition(int[] arr) {
            this.arr = arr.clone();
            int n = arr.length;
            blockSize = (int)Math.sqrt(n);
            numBlocks = (n + blockSize - 1) / blockSize;
            blocks = new int[numBlocks];
            
            for (int i = 0; i < n; i++) {
                blocks[i / blockSize] += arr[i];
            }
        }
        
        public void update(int idx, int val) {
            int blockIdx = idx / blockSize;
            blocks[blockIdx] += val - arr[idx];
            arr[idx] = val;
        }
        
        public int query(int l, int r) {
            int sum = 0;
            while (l <= r) {
                if (l % blockSize == 0 && l + blockSize - 1 <= r) {
                    sum += blocks[l / blockSize];
                    l += blockSize;
                } else {
                    sum += arr[l];
                    l++;
                }
            }
            return sum;
        }
    }
    
    // ==================== SPECIALIZED TECHNIQUES ====================
    
    /**
     * Pattern 71: Rolling Hash (Rabin-Karp)
     * Time: O(n + m)
     * Space: O(1)
     */
    static class RollingHash {
        private static final int BASE = 256;
        private static final int MOD = 1_000_000_007;
        
        public static List<Integer> rabinKarp(String text, String pattern) {
            List<Integer> matches = new ArrayList<>();
            int m = pattern.length();
            int n = text.length();
            
            long patternHash = 0;
            long textHash = 0;
            long h = 1;
            
            for (int i = 0; i < m - 1; i++) {
                h = (h * BASE) % MOD;
            }
            
            for (int i = 0; i < m; i++) {
                patternHash = (BASE * patternHash + pattern.charAt(i)) % MOD;
                textHash = (BASE * textHash + text.charAt(i)) % MOD;
            }
            
            for (int i = 0; i <= n - m; i++) {
                if (patternHash == textHash) {
                    if (text.substring(i, i + m).equals(pattern)) {
                        matches.add(i);
                    }
                }
                
                if (i < n - m) {
                    textHash = (BASE * (textHash - text.charAt(i) * h) + 
                               text.charAt(i + m)) % MOD;
                    if (textHash < 0) textHash += MOD;
                }
            }
            return matches;
        }
    }
    
    /**
     * Pattern 72: KMP Algorithm
     * Time: O(n + m)
     * Space: O(m)
     */
    static class KMPAlgorithm {
        private static int[] computeLPS(String pattern) {
            int m = pattern.length();
            int[] lps = new int[m];
            int len = 0;
            int i = 1;
            
            while (i < m) {
                if (pattern.charAt(i) == pattern.charAt(len)) {
                    len++;
                    lps[i] = len;
                    i++;
                } else {
                    if (len != 0) {
                        len = lps[len - 1];
                    } else {
                        lps[i] = 0;
                        i++;
                    }
                }
            }
            return lps;
        }
        
        public static List<Integer> kmpSearch(String text, String pattern) {
            List<Integer> matches = new ArrayList<>();
            int n = text.length();
            int m = pattern.length();
            int[] lps = computeLPS(pattern);
            
            int i = 0, j = 0;
            while (i < n) {
                if (pattern.charAt(j) == text.charAt(i)) {
                    i++;
                    j++;
                }
                
                if (j == m) {
                    matches.add(i - j);
                    j = lps[j - 1];
                } else if (i < n && pattern.charAt(j) != text.charAt(i)) {
                    if (j != 0) {
                        j = lps[j - 1];
                    } else {
                        i++;
                    }
                }
            }
            return matches;
        }
    }
    
    /**
     * Pattern 73: Z-Algorithm
     * Time: O(n)
     * Space: O(n)
     */
    static class ZAlgorithm {
        public static int[] zAlgorithm(String s) {
            int n = s.length();
            int[] z = new int[n];
            int l = 0, r = 0;
            
            for (int i = 1; i < n; i++) {
                if (i > r) {
                    l = r = i;
                    while (r < n && s.charAt(r - l) == s.charAt(r)) {
                        r++;
                    }
                    z[i] = r - l;
                    r--;
                } else {
                    int k = i - l;
                    if (z[k] < r - i + 1) {
                        z[i] = z[k];
                    } else {
                        l = i;
                        while (r < n && s.charAt(r - l) == s.charAt(r)) {
                            r++;
                        }
                        z[i] = r - l;
                        r--;
                    }
                }
            }
            return z;
        }
        
        public static List<Integer> patternSearch(String text, String pattern) {
            String s = pattern + "$" + text;
            int[] z = zAlgorithm(s);
            List<Integer> matches = new ArrayList<>();
            
            for (int i = pattern.length() + 1; i < s.length(); i++) {
                if (z[i] == pattern.length()) {
                    matches.add(i - pattern.length() - 1);
                }
            }
            return matches;
        }
    }
    
    /**
     * Pattern 74: Manacher's Algorithm
     * Time: O(n)
     * Space: O(n)
     */
    static class ManacherAlgorithm {
        public static String longestPalindrome(String s) {
            StringBuilder sb = new StringBuilder("^#");
            for (char c : s.toCharArray()) {
                sb.append(c).append('#');
            }
            sb.append(');
            String t = sb.toString();
            
            int n = t.length();
            int[] p = new int[n];
            int center = 0, right = 0;
            
            for (int i = 1; i < n - 1; i++) {
                int mirror = 2 * center - i;
                
                if (i < right) {
                    p[i] = Math.min(right - i, p[mirror]);
                }
                
                while (t.charAt(i + p[i] + 1) == t.charAt(i - p[i] - 1)) {
                    p[i]++;
                }
                
                if (i + p[i] > right) {
                    center = i;
                    right = i + p[i];
                }
            }
            
            int maxLen = 0, centerIdx = 0;
            for (int i = 1; i < n - 1; i++) {
                if (p[i] > maxLen) {
                    maxLen = p[i];
                    centerIdx = i;
                }
            }
            
            int start = (centerIdx - maxLen) / 2;
            return s.substring(start, start + maxLen);
        }
    }
    
    /**
     * Pattern 75: Reservoir Sampling
     * Time: O(n)
     * Space: O(k)
     */
    static class ReservoirSampling {
        private Random random = new Random();
        
        public List<Integer> sample(int[] stream, int k) {
            List<Integer> reservoir = new ArrayList<>();
            
            for (int i = 0; i < stream.length; i++) {
                if (i < k) {
                    reservoir.add(stream[i]);
                } else {
                    int j = random.nextInt(i + 1);
                    if (j < k) {
                        reservoir.set(j, stream[i]);
                    }
                }
            }
            return reservoir;
        }
        
        // Random pick index
        static class RandomPick {
            private int[] nums;
            private Random random;
            
            public RandomPick(int[] nums) {
                this.nums = nums;
                this.random = new Random();
            }
            
            public int pick(int target) {
                int count = 0;
                int result = -1;
                
                for (int i = 0; i < nums.length; i++) {
                    if (nums[i] == target) {
                        count++;
                        if (random.nextInt(count) == 0) {
                            result = i;
                        }
                    }
                }
                return result;
            }
        }
    }
    
    /**
     * Pattern 76: Boyer-Moore Voting Algorithm
     * Time: O(n)
     * Space: O(1)
     */
    static class BoyerMooreVoting {
        // Find majority element (> n/2)
        public static int majorityElement(int[] nums) {
            Integer candidate = null;
            int count = 0;
            
            for (int num : nums) {
                if (count == 0) {
                    candidate = num;
                }
                count += (num == candidate) ? 1 : -1;
            }
            return candidate;
        }
        
        // Find elements appearing > n/3 times
        public static List<Integer> majorityElementII(int[] nums) {
            Integer candidate1 = null, candidate2 = null;
            int count1 = 0, count2 = 0;
            
            for (int num : nums) {
                if (candidate1 != null && num == candidate1) {
                    count1++;
                } else if (candidate2 != null && num == candidate2) {
                    count2++;
                } else if (count1 == 0) {
                    candidate1 = num;
                    count1 = 1;
                } else if (count2 == 0) {
                    candidate2 = num;
                    count2 = 1;
                } else {
                    count1--;
                    count2--;
                }
            }
            
            List<Integer> result = new ArrayList<>();
            count1 = count2 = 0;
            for (int num : nums) {
                if (candidate1 != null && num == candidate1) count1++;
                if (candidate2 != null && num == candidate2) count2++;
            }
            
            if (count1 > nums.length / 3) result.add(candidate1);
            if (count2 > nums.length / 3) result.add(candidate2);
            return result;
        }
    }
    
    /**
     * Pattern 77: Game Theory / Minimax
     * Time: O(2^n) or O(n²) with memoization
     * Space: O(n)
     */
    static class GameTheory {
        // Stone game
        public static boolean stoneGame(int[] piles) {
            int n = piles.length;
            int[][] dp = new int[n][n];
            
            for (int i = 0; i < n; i++) {
                dp[i][i] = piles[i];
            }
            
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i <= n - len; i++) {
                    int j = i + len - 1;
                    dp[i][j] = Math.max(piles[i] - dp[i+1][j], 
                                       piles[j] - dp[i][j-1]);
                }
            }
            return dp[0][n-1] > 0;
        }
        
        // Predict the winner
        public static boolean PredictTheWinner(int[] nums) {
            return stoneGame(nums);
        }
    }
    
    /**
     * Pattern 78: Matrix Exponentiation
     * Time: O(k³ log n) where k is matrix dimension
     * Space: O(k²)
     */
    static class MatrixExponentiation {
        private static final int MOD = 1_000_000_007;
        
        public static long[][] multiply(long[][] A, long[][] B) {
            int n = A.length;
            long[][] C = new long[n][n];
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % MOD;
                    }
                }
            }
            return C;
        }
        
        public static long[][] power(long[][] M, int n) {
            int size = M.length;
            long[][] result = new long[size][size];
            for (int i = 0; i < size; i++) {
                result[i][i] = 1;
            }
            
            while (n > 0) {
                if (n % 2 == 1) {
                    result = multiply(result, M);
                }
                M = multiply(M, M);
                n /= 2;
            }
            return result;
        }
        
        // Fibonacci using matrix exponentiation
        public static long fibonacci(int n) {
            if (n <= 1) return n;
            
            long[][] M = {{1, 1}, {1, 0}};
            long[][] result = power(M, n - 1);
            return result[0][0];
        }
    }
    
    /**
     * Pattern 79: Meet in the Middle
     * Time: O(2^(n/2))
     * Space: O(2^(n/2))
     */
    static class MeetInTheMiddle {
        public static int subsetSum(int[] arr, int target) {
            int n = arr.length;
            int mid = n / 2;
            
            Map<Integer, Integer> leftSums = new HashMap<>();
            for (int mask = 0; mask < (1 << mid); mask++) {
                int sum = 0;
                for (int i = 0; i < mid; i++) {
                    if ((mask & (1 << i)) != 0) {
                        sum += arr[i];
                    }
                }
                leftSums.put(sum, leftSums.getOrDefault(sum, 0) + 1);
            }
            
            int count = 0;
            for (int mask = 0; mask < (1 << (n - mid)); mask++) {
                int sum = 0;
                for (int i = 0; i < n - mid; i++) {
                    if ((mask & (1 << i)) != 0) {
                        sum += arr[mid + i];
                    }
                }
                if (leftSums.containsKey(target - sum)) {
                    count += leftSums.get(target - sum);
                }
            }
            return count;
        }
    }
    
    /**
     * Pattern 82: Intervals (Merge & Operations)
     * Time: O(n log n)
     * Space: O(n)
     */
    static class IntervalOperations {
        // Merge intervals
        public static int[][] merge(int[][] intervals) {
            if (intervals.length == 0) return new int[0][0];
            
            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
            List<int[]> merged = new ArrayList<>();
            merged.add(intervals[0]);
            
            for (int i = 1; i < intervals.length; i++) {
                int[] last = merged.get(merged.size() - 1);
                if (intervals[i][0] <= last[1]) {
                    last[1] = Math.max(last[1], intervals[i][1]);
                } else {
                    merged.add(intervals[i]);
                }
            }
            return merged.toArray(new int[merged.size()][]);
        }
        
        // Interval intersection
        public static int[][] intervalIntersection(int[][] A, int[][] B) {
            List<int[]> result = new ArrayList<>();
            int i = 0, j = 0;
            
            while (i < A.length && j < B.length) {
                int start = Math.max(A[i][0], B[j][0]);
                int end = Math.min(A[i][1], B[j][1]);
                
                if (start <= end) {
                    result.add(new int[]{start, end});
                }
                
                if (A[i][1] < B[j][1]) {
                    i++;
                } else {
                    j++;
                }
            }
            return result.toArray(new int[result.size()][]);
        }
    }
    
    /**
     * Pattern 83: Custom Data Structures (LRU Cache)
     * Time: O(1) for get and put
     * Space: O(capacity)
     */
    static class LRUCache {
        class Node {
            int key, val;
            Node prev, next;
            Node(int key, int val) {
                this.key = key;
                this.val = val;
            }
        }
        
        private Map<Integer, Node> cache;
        private int capacity;
        private Node head, tail;
        
        public LRUCache(int capacity) {
            this.capacity = capacity;
            cache = new HashMap<>();
            head = new Node(0, 0);
            tail = new Node(0, 0);
            head.next = tail;
            tail.prev = head;
        }
        
        public int get(int key) {
            if (!cache.containsKey(key)) return -1;
            Node node = cache.get(key);
            remove(node);
            addToHead(node);
            return node.val;
        }
        
        public void put(int key, int value) {
            if (cache.containsKey(key)) {
                remove(cache.get(key));
            }
            Node node = new Node(key, value);
            addToHead(node);
            cache.put(key, node);
            
            if (cache.size() > capacity) {
                Node lru = tail.prev;
                remove(lru);
                cache.remove(lru.key);
            }
        }
        
        private void remove(Node node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }
        
        private void addToHead(Node node) {
            node.next = head.next;
            node.prev = head;
            head.next.prev = node;
            head.next = node;
        }
    }
    
    /**
     * Pattern 84: Expand Around Center
     * Time: O(n²)
     * Space: O(1)
     */
    static class ExpandAroundCenter {
        public static String longestPalindrome(String s) {
            if (s == null || s.length() == 0) return "";
            
            int start = 0, maxLen = 0;
            
            for (int i = 0; i < s.length(); i++) {
                int len1 = expandAroundCenter(s, i, i);
                int len2 = expandAroundCenter(s, i, i + 1);
                int len = Math.max(len1, len2);
                
                if (len > maxLen) {
                    maxLen = len;
                    start = i - (len - 1) / 2;
                }
            }
            return s.substring(start, start + maxLen);
        }
        
        private static int expandAroundCenter(String s, int left, int right) {
            while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
                left--;
                right++;
            }
            return right - left - 1;
        }
        
        // Count palindromic substrings
        public static int countSubstrings(String s) {
            int count = 0;
            for (int i = 0; i < s.length(); i++) {
                count += countPalindromes(s, i, i);
                count += countPalindromes(s, i, i + 1);
            }
            return count;
        }
        
        private static int countPalindromes(String s, int left, int right) {
            int count = 0;
            while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
                count++;
                left--;
                right++;
            }
            return count;
        }
    }
    
    /**
     * Pattern 85: Subsets Generation
     * Time: O(2^n)
     * Space: O(n)
     */
    static class SubsetGeneration {
        // Iterative with bitmask
        public static List<List<Integer>> subsetsIterative(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            int n = nums.length;
            
            for (int mask = 0; mask < (1 << n); mask++) {
                List<Integer> subset = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if ((mask & (1 << i)) != 0) {
                        subset.add(nums[i]);
                    }
                }
                result.add(subset);
            }
            return result;
        }
        
        // Backtracking
        public static List<List<Integer>> subsetsBacktrack(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            backtrack(nums, 0, new ArrayList<>(), result);
            return result;
        }
        
        private static void backtrack(int[] nums, int start, 
                                     List<Integer> path, List<List<Integer>> result) {
            result.add(new ArrayList<>(path));
            
            for (int i = start; i < nums.length; i++) {
                path.add(nums[i]);
                backtrack(nums, i + 1, path, result);
                path.remove(path.size() - 1);
            }
        }
        
        // With duplicates
        public static List<List<Integer>> subsetsWithDup(int[] nums) {
            Arrays.sort(nums);
            List<List<Integer>> result = new ArrayList<>();
            backtrackWithDup(nums, 0, new ArrayList<>(), result);
            return result;
        }
        
        private static void backtrackWithDup(int[] nums, int start,
                                            List<Integer> path, List<List<Integer>> result) {
            result.add(new ArrayList<>(path));
            for (int i = start; i < nums.length; i++) {
            // Skip duplicates: if current element is same as previous and we're not
            // at the start position, skip it to avoid duplicate subsets
            if (i > start && nums[i] == nums[i-1]) {
                continue;
            }
            path.add(nums[i]);
            backtrackWithDup(nums, i + 1, path, result);
            path.remove(path.size() - 1);
        }
    }

    // Cascading approach (iterative)
    public static List<List<Integer>> subsetsCascading(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<>()); // Start with empty subset
        
        for (int num : nums) {
            int size = result.size();
            for (int i = 0; i < size; i++) {
                List<Integer> newSubset = new ArrayList<>(result.get(i));
                newSubset.add(num);
                result.add(newSubset);
            }
        }
        return result;
    }
}
