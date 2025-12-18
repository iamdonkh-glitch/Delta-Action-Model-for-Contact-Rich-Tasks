# Delta-Action-Model-for-Contact-Rich-Tasks
Learning a delta action model that makes up for dynamic difference between sim and real for contact-rich tasks

## How to Add Videos to README

**Yes, you can add videos to your README file!** GitHub supports multiple methods for embedding videos:

### Method 1: Direct Video Upload (Recommended for GitHub)
GitHub allows you to drag and drop video files directly into issues, pull requests, and markdown files. Supported formats include `.mp4` and `.mov` files (must be under 10MB).

To add a video:
1. Drag and drop your video file into the README editor on GitHub
2. GitHub will automatically upload it and generate a URL (the exact format is generated automatically by GitHub)

### Method 2: Using HTML Video Tag
You can use HTML5 video tags in your README:

```html
<video src="https://your-video-url.mp4" controls="controls" style="max-width: 100%;">
</video>
```

### Method 3: YouTube or Vimeo Embed
GitHub doesn't support iframe embeds, so use a linked thumbnail approach instead.

**For YouTube:**
```markdown
[![Video Title](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
```
Replace `VIDEO_ID` with your actual YouTube video ID.

**For Vimeo:**
```markdown
[![Video Title](https://vumbnail.com/VIDEO_ID.jpg)](https://vimeo.com/VIDEO_ID)
```
Replace `VIDEO_ID` with your actual Vimeo video ID.

### Method 4: Animated GIFs
Convert your video to an animated GIF and include it as an image:

```markdown
![Demo](path/to/demo.gif)
```

### Method 5: Link to Video File
Simply provide a link to your video file:

```markdown
[Watch the demo video](./videos/demo.mp4)
```

### Best Practices
- Keep videos under 10MB for direct GitHub uploads
- Use descriptive alt text for accessibility
- Consider hosting longer videos on YouTube or Vimeo
- Provide captions or descriptions for better accessibility
- Store video files in a dedicated directory (e.g., `./media/` or `./videos/`)

### Example Video Section
Here's a template you can use:

```markdown
## Demo Video

Watch a demonstration of the delta action model in action:

[Link to demo video](./videos/demo.mp4)

Or view on YouTube: [Demo Video](https://youtube.com/watch?v=YOUR_VIDEO_ID)
```

## How to Upload Your Local Folder to This Repository

If you have a folder on your computer with project files and want to upload them to this repository (so the files appear at the root level, not nested in a folder), follow these steps:

### Method 1: Using Git Command Line (Recommended)

1. **Clone this repository** (if you haven't already):
   ```bash
   git clone https://github.com/iamdonkh-glitch/Delta-Action-Model-for-Contact-Rich-Tasks.git
   cd Delta-Action-Model-for-Contact-Rich-Tasks
   ```
   Or for any repository: `git clone <your-repo-url>`

2. **Copy your files** from your local folder into the cloned repository:
   ```bash
   # Copy all files from your folder to the repository root (including hidden files)
   cp -r /path/to/your/folder/. .
   # Or on Windows (Command Prompt):
   # xcopy C:\path\to\your\folder\ . /E /H /Y
   ```

3. **Add and commit your files**:
   ```bash
   git add .
   git commit -m "Add project files"
   ```

4. **Push your changes**:
   ```bash
   # Check your current branch name
   git branch --show-current
   # Then push (replace 'main' with your branch name if different, e.g., 'master')
   git push -u origin main
   ```

### Method 2: Using GitHub Desktop

1. **Clone the repository** using GitHub Desktop
2. **Open the repository folder** in your file explorer
3. **Copy all files** from your folder and paste them into the repository folder
4. **Commit and push** the changes using GitHub Desktop

### Method 3: Using GitHub Web Interface (for small projects)

1. **Navigate to your repository** on GitHub
2. Click **"Add file"** → **"Upload files"**
3. **Drag and drop** all files from your folder (not the folder itself)
4. Click **"Commit changes"**

### Important Tips

- **Don't drag the folder itself** - open the folder and drag/copy the files inside it
- **Be careful with the README.md** - if your folder has a README, it will replace this one
- **Use `.gitignore`** to exclude files you don't want to upload (like `node_modules/`, `.env`, etc.)
- **For large files** (>100MB), consider using Git LFS (Large File Storage)

### Example `.gitignore` File

Create a `.gitignore` file to exclude common files you don't want to upload:

```gitignore
# Dependencies
node_modules/
venv/
__pycache__/

# Environment variables
.env
.env.local

# IDE settings
.vscode/
.idea/
*.swp

# Build outputs
dist/
build/
*.pyc

# OS files
.DS_Store
Thumbs.db
```
