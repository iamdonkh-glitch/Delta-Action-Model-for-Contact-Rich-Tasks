# Delta-Action-Model-for-Contact-Rich-Tasks
Learning a delta action model that makes up for dynamic difference between sim and real for contact-rich tasks

## How to Add Videos to README

**Yes, you can add videos to your README file!** GitHub supports multiple methods for embedding videos:

### Method 1: Direct Video Upload (Recommended for GitHub)
GitHub allows you to drag and drop video files directly into issues, pull requests, and markdown files. Supported formats include `.mp4` and `.mov` files (must be under 10MB).

To add a video:
1. Drag and drop your video file into the README editor on GitHub
2. GitHub will automatically upload it and generate markdown like:
```markdown
https://user-images.githubusercontent.com/your-video-url.mp4
```

### Method 2: Using HTML Video Tag
You can use HTML5 video tags in your README:

```html
<video src="https://your-video-url.mp4" controls="controls" style="max-width: 730px;">
</video>
```

### Method 3: YouTube or Vimeo Embed
For YouTube videos, use a linked thumbnail approach (GitHub doesn't support iframe):

```markdown
[![Video Title](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
```

Replace `VIDEO_ID` with your actual YouTube video ID.

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
