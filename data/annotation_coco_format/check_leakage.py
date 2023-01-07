import argparse
import json

def images_set(json_path):
    with open(json_path, 'r') as f:
        ann = json.load(f)
    return set([image['file_name'] for image in ann['images']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json1', help='annotation path 1')
    parser.add_argument('--json2', help='annotation path 2')
    opt = parser.parse_args()

    duplicate_image = images_set(opt.json1).intersection(images_set(opt.json2))
    print(f"Number of duplicate image: {len(duplicate_image)}")
