# Praktikum 1: Neural Network in C++

This project implements a small neural network in C++. 

## Work plan


| Member | Tasks | Points |
|---|---|---:|
| **A ** | 3. LeakyReLU and ReLU (1); 4.`dotProduct` and `sum` (1); 6. Network constructor/destructor and full forward propagation (1.5); 7. Binary Cross-Entropy `getCostValue` (1) | 4.5 |
| **B ** | 3. ScaledSigmoid and Sigmoid (1); 4. both `normalize` functions (1); 6. Network `addLayer`, `getLayerInput`, and full backward propagation (1.5); 7. Binary Cross-Entropy accuracy and gradient (1) | 4.5 |
| **C ** | 5. Layer (4); 6. `trainBatch` and `computeAccuracy` (1) | 5 |


## Git collaboration workflow

Each person should work on their own branch. Do not commit directly to `main`. The shared branch names are:

| Member | Branch |
|---|---|
| Faran Ahmed | `faran-ahmed` |
| Anish Chand | `anish-chand` |
| Duc Anh Nguyen | `duc-anh-nguyen` |

### 1. Clone the project (once)

Open a terminal and run:

```sh
git clone https://github.com/dummy-irl/PP1.git
cd PP1
```

If GitHub asks you to authenticate when pushing, sign in with your GitHub account or configure a GitHub-supported authentication method (such as SSH or GitHub CLI).

### 2. Switch to your branch

Do this once after cloning. Replace the branch name with your own:

```sh
git fetch origin
git switch --track origin/your-name
```

Git will create a local branch that tracks the matching branch on GitHub.

Before starting work on another day, update your local branch:

```sh
git switch your-name
git pull
```

`git status` shows your current branch and any changed files.

### 3. Make and commit your changes

Edit the implementation under `code/`. From the repository root, inspect and stage only the files you intend to include:

```sh
git status
git add code/src/lib/activation/ReLU.cpp code/include/lib/activation/ReLU.h
git diff --cached
git commit -m "Implement ReLU activation"
```

Change the example file paths and commit message to match your work. You can use `git add <file>` for specific files or `git add -p` to choose individual changes. Avoid `git add .` if you have unrelated files in the working tree.

### 4. Push your branch to GitHub

```sh
git push
```

Because the branch tracks its matching `origin` branch, `git push` sends your commit there. For a newly created branch with no upstream, use `git push -u origin your-branch-name` once.

### 5. Open a pull request

On GitHub, open a pull request with your branch as the **compare** branch and `main` as the **base** branch. Describe what you changed, mention the tests you ran, and ask a teammate to review it. After review and agreement, merge the pull request into `main`.

### 6. Bring the latest `main` into your branch

After `main` has new merged work, update your feature branch before continuing or opening/updating a pull request:

```sh
git switch your-name
git fetch origin
git merge origin/main
git push
```
