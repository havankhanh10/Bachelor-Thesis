import matplotlib.pyplot as plt

def display_images(images, titles):
    num_images = len(images)
    num_cols = min(num_images, 3)  # Calculate the number of columns needed
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 10))

    # Plot the images with titles
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()