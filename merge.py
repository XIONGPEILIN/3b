from PIL import Image

# Load the images
img1 = Image.open('snake.jpg')
img2 = Image.open('peacock.jpg')
img3 = Image.open('cat.jpg')
img4 = Image.open('agama.jpg')

# Resize the images to the same size
width, height = 400, 400  # adjust the size as needed
img1 = img1.resize((width, height))
img2 = img2.resize((width, height))
img3 = img3.resize((width, height))
img4 = img4.resize((width, height))

# Create a 2x2 layout
top_row = Image.new('RGB', (width*2, height))
top_row.paste(img1, (0, 0))
top_row.paste(img2, (width, 0))

bottom_row = Image.new('RGB', (width*2, height))
bottom_row.paste(img3, (0, 0))
bottom_row.paste(img4, (width, 0))

final_image = Image.new('RGB', (width*2, height*2))
final_image.paste(top_row, (0, 0))
final_image.paste(bottom_row, (0, height))

# Save the final image
final_image.save('merged_image.jpg')