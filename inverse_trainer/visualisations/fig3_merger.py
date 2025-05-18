import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load images with the new "cropped_" prefix

se_prediction_images = [
    mpimg.imread("/Users/toprak/Desktop/cropped_landscape_0TVL.png"),
    mpimg.imread("/Users/toprak/Desktop/cropped_landscape_01TVL.png"),
    mpimg.imread("/Users/toprak/Desktop/cropped_landscape_015TVL.png"),
    mpimg.imread("/Users/toprak/Desktop/cropped_landscape_02TVL.png")
]

heatmap_images = [
    mpimg.imread("/Users/toprak/Desktop/cropped_SE_predictions_0TVL.png"),
    mpimg.imread("/Users/toprak/Desktop/cropped_SE_predictions_01TVL.png"),
    mpimg.imread("/Users/toprak/Desktop/cropped_SE_predictions_015TVL.png"),
    mpimg.imread("/Users/toprak/Desktop/cropped_SE_predictions_02TVL.png")
]

# Titles for columns
titles = ["TVK Yok", "TVK=0.1", "TVK=0.15", "TVK=0.2"]

# Create figure with removed spacing and tight layout
fig, axes = plt.subplots(2, 4, figsize=(16, 7), tight_layout=True,
                         gridspec_kw={'wspace': 0.0, 'hspace': 0.0, 'height_ratios': [1, 1]})

# Plot heatmaps on top row with smaller title font
for ax, img, title in zip(axes[0], heatmap_images, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=30)
    ax.axis('off')

# Plot SE predictions on bottom row
for ax, img in zip(axes[1], se_prediction_images):
    ax.imshow(img)
    ax.axis('off')

# Save and display
plt.savefig("merged_TVL_analysis_final.png", dpi=300, bbox_inches='tight')
plt.show()