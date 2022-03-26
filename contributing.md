# Contribution Workflow

1. When preparing to update any code **functionality** with a bug-fix, please open a GitHub issue and assign yourself and/or someone else to it.
   1. The point of doing this is to let the rest of the community know exactly who's fixing what. This proactively prevents us from running into merge conflicts when it's time to submit code into the codebase.
1. While making changes to the code, continuously check to make sure you're not breaking any tests. You can run all the tests with the command `python -m unittest discover`. You can run an individual test file with the command `python -m unittest RELATIVE_PATH_TO_TEST_FILE.py`.
1. Once you've finished making changes to the code and all the tests pass, be sure your code conforms to the style guidelines (see [below](#code-style-guide)).
1. Then, you may merge the code directly into the main branch or submit a pull request which someone will review.
1. Once your code has been merged in, close out the original issue you made by referencing the commits that have fixed the issue.

A model example of this entire workflow can be found [here](https://github.com/AI4Finance-Foundation/ElegantRL/issues/116).

# Code Style Guide

- We make use of the automatic [Black](https://black.readthedocs.io/en/stable/) Python code formatter for development speed. It should be installed with `pip`.
  - When finished updating your code files, run the command `black file.py` where `file.py` is any file you've changed.
- Functions should not be longer than ~20 lines (usually).
  - Following this general rule of thumb results in code that is easier to add to, modify, and understand. It also makes fixing bugs a lot easier.
  - If a function stretches beyond 20-30 lines, there is almost certainly a piece of logic in there that can be abstracted out into its own function - think about it!
- If a long comment would stretch beyond 88 columns, it should be wrapped over to the next line to prevent horizontal scrolling. Most modern IDEs will tell you what column your cursor is at.
- For general clean code practices, see [Uncle Bob's clean coding lessons](https://www.youtube.com/playlist?list=PLmmYSbUCWJ4x1GO839azG_BBw8rkh-zOj). The first two videos are excellent - the videos become less useful as you progress through them.
