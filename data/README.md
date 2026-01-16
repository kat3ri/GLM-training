# Data Directory

This directory contains training and evaluation data for GLM-Image.

## Structure

### Text-to-Image (T2I) Data

```
t2i/
├── prompts.txt              # Training prompts (one per line)
├── target_images/           # Target images for training
│   ├── 0.png               # Corresponds to line 0 in prompts.txt
│   ├── 1.png               # Corresponds to line 1 in prompts.txt
│   └── ...
├── eval_prompts.txt         # Evaluation prompts
└── eval_target_images/      # Target images for evaluation
    ├── 0.png
    └── ...
```

### Image-to-Image (I2I) Data

```
i2i/
├── source_images/           # Source images for editing
│   ├── 0.png
│   └── ...
├── prompts.txt              # Edit instructions (one per line)
├── target_images/           # Target edited images
│   ├── 0.png               # Result of applying prompt 0 to source 0
│   └── ...
├── eval_source_images/      # Evaluation source images
├── eval_prompts.txt         # Evaluation edit instructions
└── eval_target_images/      # Evaluation target images
```

## Image Requirements

- **Format**: PNG, JPG, JPEG, or WebP
- **Size**: Images will be resized to match the configured resolution (default: 1024x1024)
- **Resolution**: Must be divisible by 32 (e.g., 1024, 512, 2048)
- **Naming**: Use sequential numbers starting from 0: `0.png`, `1.png`, `2.png`, etc.

## Prompt Guidelines

### For T2I Training

- Be descriptive and specific
- Include quoted text if you want specific text rendered in images
- Example: `"SALE 50% OFF" poster with red and yellow colors`

### For I2I Training

- Describe the desired transformation
- Be specific about what should change and what should stay the same
- Examples:
  - `Replace the background with a beach scene`
  - `Change the sky to sunset colors`
  - `Add "Summer Vibes" text overlay`

## Creating Your Dataset

1. **Collect or generate images**: Use existing images or generate them with GLM-Image
2. **Organize files**: Name them sequentially (0.png, 1.png, etc.)
3. **Write prompts**: Create prompts.txt with one prompt per line
4. **Verify alignment**: Ensure prompt line N corresponds to image N.png

## Example Datasets

See `examples/USAGE.md` for guidance on creating datasets for specific use cases.

## Notes

- The example files in this directory are placeholders
- Replace them with your actual training data
- Evaluation data is optional but recommended for monitoring progress
- Keep a backup of your original data before training
