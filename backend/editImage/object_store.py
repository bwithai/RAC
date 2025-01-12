import os
import pickle
import pprint


# Save the object
def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Object saved to {filename}")


# Load the object
def load_object(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    print(f"Object loaded from {filename}")
    return obj


def closest_to_zero(value1, value2):
    # Round the values to 4 decimal places
    value1 = round(value1, 4)
    value2 = round(value2, 4)

    # Compare absolute values
    if abs(value1) < abs(value2):
        return value1
    else:
        return value2

def get_images_path(image_dir):
    # List to store image paths
    image_paths = []

    # Walk through the directory and get all image files
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
                image_paths.append(os.path.join(root, file))

    return image_paths

# image_paths = get_images_path("assets/demo")
# for image_path in image_paths:
#     # Get the filename and extension from the image path
#     filename = os.path.basename(image_path)
#     print(filename)
# pprint.pprint(image_paths)


# print(load_object("../database/face_obj/my_obj.pkl"))

# a = -20
# if a <= -30:
#     print('yes')
filename = 'gul.khan'
name, extnshn = filename.split('.')
print(name, extnshn)
"""
check pitch values:
tensor([[16.2186]], device='cuda:0')
greater then 10
check yaw values:
tensor([[50.5041]], device='cuda:0')
check roll values:
tensor([[12.5356]], device='cuda:0')
{{{{{{{{{{{{{{{{{{{{{{{ --- LEFT --- }}}}}}}}}}}}}}}}}}}}}}}
check pitch values:
tensor([[11.6335]], device='cuda:0')
greater then 10
check yaw values:
tensor([[33.6694]], device='cuda:0')
check roll values:
tensor([[6.8719]], device='cuda:0')
{{{{{{{{{{{{{{{{{{{{{{{ --- LEFT --- }}}}}}}}}}}}}}}}}}}}}}}

most bad eyebrow:
eye_brow:
tensor(-0.0049, device='cuda:0')
tensor(0.0032, device='cuda:0')
tensor(-0.0046, device='cuda:0')
tensor(0.0291, device='cuda:0')
"""