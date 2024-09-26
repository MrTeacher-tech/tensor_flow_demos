import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import mimetypes

CLASS_LABEL = ""

# Fetch the page content
url = ""
response = requests.get(url)

# Parse the content with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Create a folder called 'indian_food' if it doesn't exist
folder_name = "./" + CLASS_LABEL
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Find all the <img> tags
images = soup.find_all('img')

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.webp']

# Loop through all image tags and download the images
for img in images:
    img_url = img.get('src')  # Get the 'src' attribute of the <img> tag
    if img_url:
        # Some image URLs might be relative, so we convert them to absolute URLs
        img_url = urljoin(url, img_url)

        # Check the image content type before downloading to avoid saving SVGs and JSONs
        try:
            head_response = requests.head(img_url)
            content_type = head_response.headers['content-type']
            
            # Skip SVG and JSON files
            if content_type in ['image/svg+xml', 'application/json', 'image/gif']:
                print(f"Skipping file (SVG or JSON): {img_url}")
                continue

            # Get the image content and save it
            img_data = requests.get(img_url).content

            # Extract the file name
            img_name = os.path.join(folder_name, img_url.split('/')[-1])

            # Check if the file name already has an extension
            if not any(img_name.lower().endswith(ext) for ext in image_extensions):
                ext = mimetypes.guess_extension(content_type)
                if ext:  # If an extension is found, append it to the filename
                    img_name += ext

            # Save the image
            with open(img_name, 'wb') as img_file:
                img_file.write(img_data)
            print(f"Downloaded {img_name}")
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

print("All non-SVG and non-JSON images downloaded!")
