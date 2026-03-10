# ConFu project page

Static GitHub Pages template for:

**The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment**

## Files

- `index.html` — main page
- `styles.css` — styling
- `assets/` — put teaser figures, plots, poster PDF, and supplementary files here

## Quick local preview

### Option 1: Python built-in server

```bash
cd confu-project-page
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

### Option 2: VS Code Live Server

If you use VS Code, install the **Live Server** extension, open the folder, then right-click `index.html` and choose **Open with Live Server**.

## GitHub Pages deployment

1. Create a GitHub repository.
2. Upload the files in this folder.
3. In GitHub, go to **Settings → Pages**.
4. Under **Build and deployment**, choose:
   - **Source**: Deploy from a branch
   - **Branch**: `main`
   - **Folder**: `/ (root)`
5. Save.

Your page will be published at a URL like:

```text
https://YOUR-USERNAME.github.io/YOUR-REPOSITORY/
```
