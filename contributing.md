# Contribution Workflow

1. When preparing to update any code **functionality** with a bug-fix, please open a GitHub issue and assign yourself and/or someone else to it.
   1. The point of doing this is to let the rest of the community know exactly who's fixing what. This proactively prevents us from running into merge conflicts when it's time to submit code into the codebase.
1. While making changes to the code, continuously check to make sure you're not breaking any tests. You can run the tests with the command `python -m unittest discover`.
1. Once you've finished making changes to the code and all the tests pass, be sure your code conforms to the style guidelines (see [below](#code-style-guide)).
1. Then, you may merge the code directly into the main branch or submit a pull request which someone will review.
1. Once your code has been merged in, close out the original issue you made by referencing the commits that have fixed the issue.

A model example of this entire workflow can be found [here](https://github.com/AI4Finance-Foundation/ElegantRL/issues/116).

# Code Style Guide

- We make use of the automatic [Black](https://black.readthedocs.io/en/stable/) Python code formatter for development speed. It should be installed with `pip`.
  - When finished updating your code files, run the command `black file.py` where `file.py` is any file you've changed.
- Lines should be shorter than 88 columns to prevent horizontal scrolling. Most modern IDEs will tell you what column your cursor is at.
  - If a long comment would stretch beyond 88 columns, it should be wrapped over to the next line.
  - If a string would stretch beyond 88 columns, it should be rewritten as the sum of multiple strings in Python.
