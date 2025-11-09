# CellARC Prompt Example

The following prompt mirrors the structure we use when querying LLMs for
CellARC. Each task contains several 1D training examples followed by a test
input. The model must infer the rule that maps each input sequence to its output
sequence and then predict the output for the test input. The final answer must
be a single line of space-separated integers.

```
Find the common rule that maps an input sequence to an output sequence, given
the examples below.

Example 1:

Input:
2 0 0 0 1
Output:
3 1 2 0 1

Example 2:

Input:
2 2 2 3 2
Output:
1 2 1 1 2

Example 3:

Input:
3 1 1 2 1
Output:
2 1 3 3 3

Example 4:

Input:
1 1 3 2 3
Output:
3 1 2 3 1

Example 5:

Input:
1 1 2 3 3
Output:
3 3 2 3 1

Below is a test input sequence. Predict the corresponding output sequence by
applying the rule you found. Your final answer should be a single line
containing the predicted output sequence.

Input:
1 2 2 2 1
```
