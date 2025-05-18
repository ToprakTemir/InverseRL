from PIL import Image, ImageChops
import os

# List all the images you want to trim

dir = "/Users/toprak/Desktop/"
image_filenames = [
    "SE_predictions_0TVL.png",
    "SE_predictions_01TVL.png",
    "SE_predictions_015TVL.png",
    "SE_predictions_02TVL.png",
    "landscape_0TVL.png",
    "landscape_01TVL.png",
    "landscape_015TVL.png",
    "landscape_02TVL.png",
]

for filename in image_filenames:
    # Open the image
    img = Image.open(dir + filename).convert("RGB")  # Convert to RGB just in case

    # Create a white background image of the same size
    bg = Image.new("RGB", img.size, (255, 255, 255))

    # Compute the difference
    diff = ImageChops.difference(img, bg)
    # Get the bounding box of non-zero regions in the difference
    bbox = diff.getbbox()

    if bbox:
        # Crop the image to that bounding box
        cropped_img = img.crop(bbox)
    else:
        # If bbox is None, it means the image is completely white
        # or something unexpected. In that case, just keep it as is.
        cropped_img = img

    # Construct the output filename
    # e.g., "cropped_SE_predictions_0TVL.png"
    out_filename = dir + f"cropped_{filename}"

    # Save the cropped image
    cropped_img.save(out_filename)

    print(f"Trimmed and saved: {out_filename}")
