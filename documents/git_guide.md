GIT / Installation / Usage / Best practice
https://git-scm.com/
j_smit

# Git Installation & Usage Best Practices

## 📦 Installation

### Windows
1. Download Git from the official site: https://git-scm.com/
2. Run the installer and choose the following recommended options:
   - Use Git from the Windows Command Prompt
   - Checkout Windows-style, commit Unix-style line endings
   - Use Windows' default console window

### macOS
```bash
brew install git
```

### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install git
```

### Verify Installation
```bash
git --version
```

---

## ✅ Best Practices

### 1. Set Up Identity
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 2. Use Meaningful Commit Messages
Use clear, concise messages that explain *what* and *why*, not just *how*.

```bash
git commit -m "Fix login bug by revalidating user token"
```

### 3. Branching Strategy
- Use `main` or `master` as the stable branch
- Create feature branches: `feature/login-ui`
- Use `develop` or `staging` branches for integration

### 4. Pull Before You Push
```bash
git pull origin main
```
Avoid conflicts by syncing before pushing changes.

### 5. Ignore Unnecessary Files
Use `.gitignore` to exclude build artifacts, secrets, and system files.

### 6. Use Tags for Releases
```bash
git tag -a v1.0 -m "Initial release"
```

### 7. Review with Pull Requests (PRs)
Encourage code reviews using GitHub/GitLab pull requests for collaborative work.

---

## ⚠️ Common Problems & Solutions

### 🔄 Merge Conflicts
**Problem:** Conflicting changes in the same lines of code.

**Solution:**
- Run `git status` to identify conflicted files
- Open the file and manually resolve markers:
  ```
  <<<<<<< HEAD
  Your change
  =======
  Other change
  >>>>>>> branch-name
  ```
- After resolving:
  ```bash
  git add .
  git commit
  ```

### 🚫 Detached HEAD
**Problem:** You checked out a commit instead of a branch.

**Solution:**
```bash
git checkout main
```

### 🔒 Permission Denied (SSH/HTTPS)
**Problem:** Push fails due to authentication.

**Solution:**
- For HTTPS: Use Personal Access Tokens (PAT) for GitHub
- For SSH:
  ```bash
  ssh-keygen -t ed25519 -C "you@example.com"
  ssh-add ~/.ssh/id_ed25519
  ```
  Add the public key to your GitHub/GitLab profile

### 🧼 Accidental Commit of Secrets
**Problem:** API keys or secrets are committed

**Solution:**
- Remove the secret:
  ```bash
  git rm --cached secret.txt
  git commit --amend
  ```
- Optionally rewrite history using `git filter-branch` or `BFG Repo-Cleaner`

---

## 🛠 Recommended Tools
- [GitHub CLI](https://cli.github.com/)
- [GitLens for VSCode](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
- [GitKraken](https://www.gitkraken.com/)

---

## 📚 Further Reading
- https://git-scm.com/book/en/v2
- https://docs.github.com/en/get-started/quickstart
- https://www.atlassian.com/git/tutorials