#!/usr/bin/env python3
import urllib.request

def test_url(url, name):
    try:
        response = urllib.request.urlopen(url)
        size_mb = int(response.headers.get("Content-Length", 0)) / 1024 / 1024
        print(f"✅ {name}: Available (Size: {size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ {name}: Not available ({e})")
        return False

print("Testing Oxford 102 Flowers Dataset Availability...")
print("-" * 50)

urls = [
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", "Main dataset"),
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", "Labels"),
    ("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat", "Splits")
]

all_available = True
for url, name in urls:
    if not test_url(url, name):
        all_available = False

print("-" * 50)
if all_available:
    print("🎉 All dataset files are available for download!")
    print("Total download size: ~344 MB")
    print("Extracted size: ~700-800 MB")
else:
    print("⚠️  Some dataset files are not available.")