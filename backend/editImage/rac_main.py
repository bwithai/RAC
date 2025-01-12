import os

import cv2

from editImage.object_store import get_images_path
from editImage.src.config.argument_config import ArgumentConfig
from editImage.src.config.inference_config import InferenceConfig
from editImage.src.config.crop_config import CropConfig
from editImage.rac_pipeline import RacPipeline

# Initialize configurations
inference_cfg: InferenceConfig = InferenceConfig()
crop_cfg: CropConfig = CropConfig()
args: ArgumentConfig = ArgumentConfig()

# Initialize pipeline
gradio_pipeline = RacPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)

# kami eyes x = x-eye1:  tensor(-0.0050, device='cuda:0')
# x-eye2:  tensor(-0.0041, device='cuda:0')

# Input image path
# image_data = "./assets/bad_cap/sb7.jpg"  # yaw: tensor([[20.6496]], device='cuda:0')
# image_data = "./assets/smile/ce1.jpg"
# image_data = "./assets/final/tt2.jpg"
# image_data = "./assets/final/right/rr9.jpg"
# image_data = "./assets/t_massod/cry_cry.png"
# image_data = "./assets/pose3.jpg" # yaw: tensor([[3.5409]], device='cuda:0')
# image_data = "./assets/demo/zulq.jpg"  # yaw: tensor([[3.5409]], device='cuda:0')
# image_dir = "./assets/demo"
image_dir = "../../imageDB/source"

smile = 1
mouth_ratio = 0.08


def retake_image(image_data, output_file_path, smile, lip_false=False):
    eye_retargeting_slider, lip_retargeting_slider = gradio_pipeline.init_retargeting_image(
        retargeting_source_scale=3.2,
        source_eye_ratio=0.3,
        source_lip_ratio=0.3,
        input_image=image_data)
    print('eye ', eye_retargeting_slider)
    print('lip ', lip_retargeting_slider)
    if eye_retargeting_slider > 0.4:
        input_eye_ratio = eye_retargeting_slider
    else:
        input_eye_ratio = eye_retargeting_slider + 0.01 if eye_retargeting_slider > 0.3 else 0.3998
    if lip_false:
        input_lib_ratio = 0.0
    elif lip_retargeting_slider > 0.4:
        input_lib_ratio = 0.2
    else:
        input_lib_ratio = lip_retargeting_slider if lip_retargeting_slider <= 0.1 else 0.0

    out, out_to_ori_blend = gradio_pipeline.execute_image_retargeting(
        input_eye_ratio=input_eye_ratio,
        input_lip_ratio=input_lib_ratio,
        input_head_pitch_variation=0,
        input_head_yaw_variation=0,
        input_head_roll_variation=0,
        mov_x=0.0,
        mov_y=0.0,
        mov_z=1.0,
        lip_variation_zero=0,
        lip_variation_one=0,
        lip_variation_two=0,
        lip_variation_three=0,
        smile=smile,  # (0, 0.8) (attitude, romantic)
        wink=0,
        eyebrow=0,  # (-12.0, 10) (attitude, romantic)
        eyeball_direction_x=0,  # Moving right
        eyeball_direction_y=0,  # No vertical movement
        input_image=image_data,  # Filepath directly
        retargeting_source_scale=2.5,
        flag_stitching_retargeting_input=True,
        flag_do_crop_input_retargeting_image=True
    )

    # Save the output image
    output_image = cv2.cvtColor(out_to_ori_blend, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_file_path, output_image)
    print("Output saved...")
    # input()
    # return "Output saved to 'test_sb7.jpg'"


# for i in range(2):
#     if i == 1:
#         retake_image("./assets/test.jpg", 0.2, 1.3)
#     retake_image(image_data, 0.0, -0.3)


# retake_image(image_data, './assets/test.jpg', -0.3, lip_false=True)

image_paths = get_images_path(image_dir)
for image_path in image_paths:
    filename = os.path.basename(image_path)
    name, extnshn = filename.split('.')
    simple_file_path = os.path.join("../../imageDB/output", f"{name}_simple.{extnshn}")
    smile_file_path = os.path.join("../../imageDB/output", f"{name}_smile.{extnshn}")
    close_mouth_file_path = os.path.join("../../imageDB/output", f"{name}_close_mouth.{extnshn}")
    retake_image(image_path, simple_file_path, -0.3)
    retake_image(image_path, close_mouth_file_path, -0.1, lip_false=True)
    retake_image(image_path, smile_file_path, 0.9)
    # Get the filename and extension from the image path
print("Output saved to 'assets/simple_output'")
