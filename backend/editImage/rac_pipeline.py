import torch
import pprint

from torch import tensor

from editImage.src.config.argument_config import ArgumentConfig
from editImage.live_portrait_pipeline import LivePortraitPipeline
from editImage.src.utils.io import load_img_online
from editImage.src.utils.rprint import rlog as log
from editImage.src.utils.crop import prepare_paste_back, paste_back
from editImage.src.utils.camera import get_rotation_matrix
from editImage.src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from editImage.object_store import save_object, load_object, closest_to_zero

# obj_db_path = os.path.join(os.path.dirname(database.__file__), 'face_obj')


def update_args(args, user_args):
    """update the args according to user inputs
    """
    for k, v in user_args.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


class RacPipeline(LivePortraitPipeline):
    """gradio for human
    """

    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        super().__init__(inference_cfg, crop_cfg)
        # self.live_portrait_wrapper = self.live_portrait_wrapper
        self.args = args

    @torch.no_grad()
    def update_delta_new_eyeball_direction(self, eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs):
        if eyeball_direction_x > 0:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
            delta_new[0, 15, 0] += eyeball_direction_x * 0.001
        else:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.001
            delta_new[0, 15, 0] += eyeball_direction_x * 0.0007

        delta_new[0, 11, 1] += eyeball_direction_y * -0.001
        delta_new[0, 15, 1] += eyeball_direction_y * -0.001
        # blink = -eyeball_direction_y / 2.
        #
        # delta_new[0, 11, 1] += blink * -0.001
        # delta_new[0, 13, 1] += blink * 0.0003
        # delta_new[0, 15, 1] += blink * -0.001
        # delta_new[0, 16, 1] += blink * 0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_smile(self, smile, delta_new, **kwargs):
        delta_new[0, 20, 1] += smile * -0.01
        delta_new[0, 14, 1] += smile * -0.02
        delta_new[0, 17, 1] += smile * 0.0065
        delta_new[0, 17, 2] += smile * 0.003
        delta_new[0, 13, 1] += smile * -0.00275
        delta_new[0, 16, 1] += smile * -0.00275
        delta_new[0, 3, 1] += smile * -0.0035
        delta_new[0, 7, 1] += smile * -0.0035

        return delta_new

    @torch.no_grad()
    def update_delta_new_wink(self, wink, delta_new, **kwargs):
        delta_new[0, 11, 1] += wink * 0.001
        delta_new[0, 13, 1] += wink * -0.0003
        delta_new[0, 17, 0] += wink * 0.0003
        delta_new[0, 17, 1] += wink * 0.0003
        delta_new[0, 3, 1] += wink * -0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_eyebrow(self, eyebrow, delta_new, **kwargs):
        if eyebrow > 0:
            delta_new[0, 1, 1] += eyebrow * 0.001
            delta_new[0, 2, 1] += eyebrow * -0.001
        else:
            delta_new[0, 1, 0] += eyebrow * -0.001
            delta_new[0, 2, 0] += eyebrow * 0.001
            delta_new[0, 1, 1] += eyebrow * 0.0003
            delta_new[0, 2, 1] += eyebrow * -0.0003
        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_zero(self, lip_variation_zero, delta_new, **kwargs):
        delta_new[0, 19, 0] += lip_variation_zero

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_one(self, lip_variation_one, delta_new, **kwargs):
        delta_new[0, 14, 1] += lip_variation_one * 0.001
        delta_new[0, 3, 1] += lip_variation_one * -0.0005
        delta_new[0, 7, 1] += lip_variation_one * -0.0005
        delta_new[0, 17, 2] += lip_variation_one * -0.0005

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_two(self, lip_variation_two, delta_new, **kwargs):
        delta_new[0, 20, 2] += lip_variation_two * -0.001
        delta_new[0, 20, 1] += lip_variation_two * -0.001
        delta_new[0, 14, 1] += lip_variation_two * -0.001

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_three(self, lip_variation_three, delta_new, **kwargs):
        delta_new[0, 19, 1] += lip_variation_three * 0.001
        delta_new[0, 19, 2] += lip_variation_three * 0.0001
        delta_new[0, 17, 1] += lip_variation_three * -0.0001

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_x(self, mov_x, delta_new, **kwargs):
        delta_new[0, 5, 0] += mov_x

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_y(self, mov_y, delta_new, **kwargs):
        delta_new[0, 5, 1] += mov_y

        return delta_new

    @torch.no_grad()
    def execute_image_retargeting(
            self,
            input_eye_ratio: float,
            input_lip_ratio: float,
            input_head_pitch_variation: float,
            input_head_yaw_variation: float,
            input_head_roll_variation: float,
            mov_x: float,
            mov_y: float,
            mov_z: float,
            lip_variation_zero: float,
            lip_variation_one: float,
            lip_variation_two: float,
            lip_variation_three: float,
            smile: float,
            wink: float,
            eyebrow: float,
            eyeball_direction_x: float,
            eyeball_direction_y: float,
            input_image,
            retargeting_source_scale: float,
            flag_stitching_retargeting_input=True,
            flag_do_crop_input_retargeting_image=True):
        """ for single image retargeting
        """
        if input_head_pitch_variation is None or input_head_yaw_variation is None or input_head_roll_variation is None:
            raise ValueError("Invalid relative pose input 💥!", duration=5)
        # disposable feature
        x_eye, x_d_info_user_yaw, x_d_info_user_pitch, x_d_info_user_roll, f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
            self.prepare_retargeting_image(
                input_image, input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation,
                retargeting_source_scale, flag_do_crop=flag_do_crop_input_retargeting_image)

        yaw = x_d_info_user_yaw.item()
        pitch = x_d_info_user_pitch.item()
        roll = x_d_info_user_roll.item()
        variations = {}
        print("lip is: ", input_lip_ratio)
        if yaw < -8:
            if yaw <= -30:
                if pitch > 0:
                    cstm_pitch = -4.5851
                else:
                    cstm_pitch = 4.5851
                if roll > 0:
                    cstm_roll = -5.6637
                else:
                    if roll < -10:
                        cstm_roll = 8.6872
                    else:
                        cstm_roll = 4.6872
                variations = {
                    "yaw": abs(round(yaw / 3, 4)),
                    "pitch": cstm_pitch,
                    "roll": cstm_roll,
                }
                eyeball_direction_x, eyeball_direction_y = -8.7813, 5
            elif -30 <= yaw < -20:
                variations = {
                    "yaw": 20.9404,
                    "pitch": 2.5851,
                    "roll": -5.6637,
                }
                eyeball_direction_x, eyeball_direction_y = 0.0, 0.0
            elif -20 <= yaw < -10:
                if roll > 10:
                    cstm_roll = -4.6637
                elif 2.9858 < roll <= 8:
                    cstm_roll = 4.6637
                else:
                    cstm_roll = 0.0
                variations = {
                    "yaw": 6.9404,
                    "pitch": 2.5851,
                    "roll": cstm_roll,
                }
                eyeball_direction_x, eyeball_direction_y = -7.0, 0.0
            else:
                variations = {
                    "yaw": 4.7813,
                    "pitch": -3.5851,
                    "roll": -0.6637,
                }
                eyeball_direction_x, eyeball_direction_y =4.7813, 0.0
        elif yaw > 8:
            if yaw >= 30:
                if pitch > 0:
                    cstm_pitch = -4.5851
                else:
                    cstm_pitch = 4.5851
                if roll > 0:
                    cstm_roll = -5.6637
                else:
                    if roll < -10:
                        cstm_roll = 8.6872
                    else:
                        cstm_roll = 4.6872
                variations = {
                    "yaw": -round(yaw / 3, 4),
                    "pitch": cstm_pitch,
                    "roll": cstm_roll,
                }
                eyeball_direction_x, eyeball_direction_y = 8.7813, 5
            elif 20 <= yaw < 30:
                variations = {
                    "yaw": -20.9404,
                    "pitch": 2.5851,
                    "roll": -5.6637,
                }
                eyeball_direction_x, eyeball_direction_y = 0.0, 0.0
            elif 10 <= yaw < 20:
                if roll > 10:
                    cstm_roll = -4.6637
                elif 2.9858 < roll <= 8:
                    cstm_roll = 4.6637
                else:
                    cstm_roll = 0.0
                # y_eye = 0.0
                # if pitch + 2.5851 > 9.9998:
                #     y_eye = - 3.0
                variations = {
                    "yaw": -6.9404,
                    "pitch": 2.5851,
                    "roll": cstm_roll,
                }
                eyeball_direction_x, eyeball_direction_y = 7.0, 0.0
            else:
                variations = {
                    "yaw": -4.7813,
                    "pitch": -3.5851,
                    "roll": -0.6637,
                }
                eyeball_direction_x, eyeball_direction_y =-4.7813, 0.0
        elif -8 <= yaw <= 8:
            if yaw > 0:
                cstm_yaw = -round(yaw, 4)
            else:
                cstm_yaw = abs(round(yaw, 4))
            if pitch > 0:
                if pitch >= 10:
                    cstm_pitch = 0.0
                else:
                    cstm_pitch = -round(pitch, 4)
            else:
                cstm_pitch = abs(round(pitch, 4))
            if roll > 0:
                cstm_roll = -round(roll, 4)
            else:
                cstm_roll = abs(round(roll, 4))
            print("////////////////////////////////////")
            print(cstm_yaw, cstm_pitch, cstm_roll)
            variations = {
                "yaw": cstm_yaw,
                "pitch": cstm_pitch,
                "roll": cstm_roll,
            }
            eyeball_direction_x, eyeball_direction_y = -4.7813, 0.0

        if variations:
            # Update retargeting parameters
            x_eye, x_d_info_user_yaw, x_d_info_user_pitch, x_d_info_user_roll, f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
                self.prepare_retargeting_image(
                    input_image, variations['pitch'], variations['yaw'], variations['roll'],
                    retargeting_source_scale, flag_do_crop=flag_do_crop_input_retargeting_image)

        if input_eye_ratio is None or input_lip_ratio is None:
            raise ValueError("Invalid ratio input 💥!", duration=5)
        else:
            device = self.live_portrait_wrapper.device
            # inference_cfg = self.live_portrait_wrapper.inference_cfg
            x_s_user = x_s_user.to(device)
            f_s_user = f_s_user.to(device)
            R_s_user = R_s_user.to(device)
            R_d_user = R_d_user.to(device)
            mov_x = torch.tensor(mov_x).to(device)
            mov_y = torch.tensor(mov_y).to(device)
            mov_z = torch.tensor(mov_z).to(device)
            eyeball_direction_x = torch.tensor(eyeball_direction_x).to(device)
            eyeball_direction_y = torch.tensor(eyeball_direction_y).to(device)
            smile = torch.tensor(smile).to(device)
            wink = torch.tensor(wink).to(device)
            eyebrow = torch.tensor(eyebrow).to(device)
            lip_variation_zero = torch.tensor(lip_variation_zero).to(device)
            lip_variation_one = torch.tensor(lip_variation_one).to(device)
            lip_variation_two = torch.tensor(lip_variation_two).to(device)
            lip_variation_three = torch.tensor(lip_variation_three).to(device)

            x_c_s = x_s_info['kp'].to(device)
            delta_new = x_s_info['exp'].to(device)
            # print("check_delta:")
            # save_object(delta_new, f"{obj_db_path}/super_man_left.pkl")
            # return
            # pprint.pprint(delta_new)
            # yaw = x_d_info_user_yaw.item()
            # if yaw < -8:
            #     delta_new = load_object(f"{obj_db_path}/super_man_right.pkl")
            # elif yaw > 8:
            #     delta_new = load_object(f"{obj_db_path}/super_man_left.pkl")
            # elif yaw >= -8 and yaw <= 8:
            #     delta_new = load_object(f"{obj_db_path}/super_man_strait.pkl")
            scale_new = x_s_info['scale'].to(device)
            t_new = x_s_info['t'].to(device)
            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

            print("before")
            # eyeball_direction_x = x_d_info_user_yaw.item()
            # eyeball_direction_y = x_d_info_user_pitch.item()
            # print(eyeball_direction_x, eyeball_direction_y)
            print("x-eye1: ", delta_new[0, 11, 0])
            print("x-eye2: ", delta_new[0, 15, 0])
            print("y-eye1: ", delta_new[0, 11, 1])
            print("y-eye2: ", delta_new[0, 15, 1])
            if yaw < 40:
                delta_new[0, 11, 0] = tensor(0.0, device='cuda:0')
                delta_new[0, 15, 0] = tensor(0.0, device='cuda:0')

            if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                delta_new = self.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
            # x_eye1 = delta_new[0, 11, 0].item()
            # x_eye2 = delta_new[0, 15, 0].item()
            # y_eye1 = delta_new[0, 11, 1].item()
            # y_eye2 = delta_new[0, 15, 1].item()
            # x_eye = closest_to_zero(x_eye1, x_eye2)
            # y_eye = closest_to_zero(y_eye1, y_eye2)

            # delta_new[0, 11, 1] = tensor(y_eye, device='cuda:0') # left eye
            # delta_new[0, 15, 1] = tensor(y_eye, device='cuda:0') # right eye
            print("after")
            print("x-eye1: ", delta_new[0, 11, 0])
            print("x-eye2: ", delta_new[0, 15, 0])
            print("y-eye1: ", delta_new[0, 11, 1])
            print("y-eye2: ", delta_new[0, 15, 1])
            if smile != 0:
                delta_new = self.update_delta_new_smile(smile, delta_new)
            print('+' * 10)
            print("smile:")
            print(delta_new[0, 20, 1], delta_new[0, 14, 1], delta_new[0, 17, 1], delta_new[0, 17, 2],
                  delta_new[0, 13, 1], delta_new[0, 16, 1], delta_new[0, 3, 1], delta_new[0, 7, 1])
            print('+' * 10)
            if wink != 0:
                delta_new = self.update_delta_new_wink(wink, delta_new)
            print('..' * 10)
            print("wink:")
            print(delta_new[0, 11, 1], delta_new[0, 13, 1], delta_new[0, 17, 0], delta_new[0, 17, 1],
                  delta_new[0, 3, 1])
            print(".." * 10)
            if eyebrow != 0:
                delta_new = self.update_delta_new_eyebrow(eyebrow, delta_new)
            print('-' * 10)
            print("eye_brow:")
            print(delta_new[0, 1, 0])
            print(delta_new[0, 2, 0])
            print(delta_new[0, 1, 1])
            print(delta_new[0, 2, 1])
            # delta_new[0, 1, 0] = tensor(0.0013, device='cuda:0')
            # delta_new[0, 2, 0] = tensor(-0.0009, device='cuda:0')
            # delta_new[0, 1, 1] = tensor(-0.0036, device='cuda:0')
            # delta_new[0, 2, 1] = tensor(0.0030, device='cuda:0')
            print('-' * 10)
            # delta_new[0, 1, 0] = tensor(0.0013, device='cuda:0')
            # delta_new[0, 2, 0] = tensor(-0.0009, device='cuda:0')
            # delta_new[0, 1, 1] = tensor(-0.0036, device='cuda:0')
            # delta_new[0, 2, 1] = tensor(0.0030, device='cuda:0')
            if lip_variation_zero != 0:
                delta_new = self.update_delta_new_lip_variation_zero(lip_variation_zero, delta_new)
            if lip_variation_one != 0:
                delta_new = self.update_delta_new_lip_variation_one(lip_variation_one, delta_new)
            if lip_variation_two != 0:
                delta_new = self.update_delta_new_lip_variation_two(lip_variation_two, delta_new)
            if lip_variation_three != 0:
                delta_new = self.update_delta_new_lip_variation_three(lip_variation_three, delta_new)
            if mov_x != 0:
                delta_new = self.update_delta_new_mov_x(-mov_x, delta_new)
            if mov_y != 0:
                delta_new = self.update_delta_new_mov_y(mov_y, delta_new)

            x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            eyes_delta, lip_delta = None, None
            if input_eye_ratio != self.source_eye_ratio:
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(
                    [[float(input_eye_ratio)]], source_lmk_user)
                eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s_user, combined_eye_ratio_tensor)
            if input_lip_ratio != self.source_lip_ratio:
                combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(
                    [[float(input_lip_ratio)]], source_lmk_user)
                lip_delta = self.live_portrait_wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor)
            x_d_new = x_d_new + \
                      (eyes_delta if eyes_delta is not None else 0) + \
                      (lip_delta if lip_delta is not None else 0)

            if flag_stitching_retargeting_input:
                x_d_new = self.live_portrait_wrapper.stitching(x_s_user, x_d_new)
            out = self.live_portrait_wrapper.warp_decode(f_s_user, x_s_user, x_d_new)
            out = self.live_portrait_wrapper.parse_output(out['out'])[0]
            if flag_do_crop_input_retargeting_image:
                out_to_ori_blend = paste_back(out, crop_M_c2o, img_rgb, mask_ori)
            else:
                out_to_ori_blend = out
            return out, out_to_ori_blend

    @torch.no_grad()
    def prepare_retargeting_image(
            self,
            input_image,
            input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation,
            retargeting_source_scale,
            flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is not None:
            # gr.Info("Upload successfully!", duration=2)
            args_user = {'scale': retargeting_source_scale}
            self.args = update_args(self.args, args_user)
            self.cropper.update_config(self.args.__dict__)
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source portrait ########
            img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=2)
            if flag_do_crop:
                crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
                I_s = self.live_portrait_wrapper.prepare_source(crop_info['img_crop_256x256'])
                source_lmk_user = crop_info['lmk_crop']
                crop_M_c2o = crop_info['M_c2o']
                mask_ori = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'],
                                              dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            else:
                I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
                source_lmk_user = self.cropper.calc_lmk_from_cropped_image(img_rgb)
                crop_M_c2o = None
                mask_ori = None
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_d_info_user_pitch = x_s_info['pitch'] + input_head_pitch_variation
            print("check pitch values:")
            pprint.pprint(x_d_info_user_pitch)
            x_d_info_user_yaw = x_s_info['yaw'] + input_head_yaw_variation
            if x_d_info_user_yaw > 10:
                print("greater then 10")
            print("check yaw values:")
            pprint.pprint(x_d_info_user_yaw)
            x_d_info_user_roll = x_s_info['roll'] + input_head_roll_variation
            print("check roll values:")
            pprint.pprint(x_d_info_user_roll)
            orignal_yaw = x_d_info_user_yaw
            yaw = x_d_info_user_yaw.item()
            x_eye = 0
            if yaw < -8:  # right
                print("{{{{{{{{{{{{{{{{{{{{{{{ --- RIGHT --- }}}}}}}}}}}}}}}}}}}}}}}")
                # x_eye = 4.7813
                # x_d_info_user_yaw = tensor([[-13.7525]], device='cuda:0')
                # x_d_info_user_pitch = tensor([[-1.9179]], device='cuda:0')
                # x_d_info_user_roll = tensor([[1.0644]], device='cuda:0')
            elif yaw > 8:  # left
                # if yaw > 20:
                #     x_eye = 14.7813
                #     x_d_info_user_yaw = tensor([[round(yaw/3, 4)]], device='cuda:0')
                #     print("Now the Yaw is: ", round(yaw/2, 4), " and eye is: ", x_eye)
                #     # x_d_info_user_pitch = tensor([[3.5851]], device='cuda:0')
                #     # x_d_info_user_roll = tensor([[0.6637]], device='cuda:0')
                # else:
                #     x_eye = -4.7813
                #     x_d_info_user_yaw = tensor([[4.7813]], device='cuda:0')
                #     x_d_info_user_pitch = tensor([[3.5851]], device='cuda:0')
                #     x_d_info_user_roll = tensor([[0.6637]], device='cuda:0')
                print("{{{{{{{{{{{{{{{{{{{{{{{ --- LEFT --- }}}}}}}}}}}}}}}}}}}}}}}")
            elif -8 <= yaw <= 8:  # strait
                print("{{{{{{{{{{{{{{{{{{{{{{{ --- STRAIT --- }}}}}}}}}}}}}}}}}}}}}}}")
                # x_d_info_user_yaw = tensor([[0.0]], device='cuda:0')
                if x_d_info_user_pitch.item() >= 10:
                    x_d_info_user_pitch = tensor([[8.6872]], device='cuda:0')
                # x_d_info_user_roll = tensor([[0.0]], device='cuda:0')
            # x_d_info_user_yaw = tensor([[-1.2858]], device='cuda:0')
            # x_d_info_user_pitch = tensor([[1.5300]], device='cuda:0')
            # x_d_info_user_roll = tensor([[3.0097]], device='cuda:0')
            R_s_user = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
            ############################################
            f_s_user = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s_user = self.live_portrait_wrapper.transform_keypoint(x_s_info)
            return x_eye, orignal_yaw, x_d_info_user_pitch, x_d_info_user_roll, f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
        else:
            raise ValueError("Please upload a source portrait as the retargeting input 🤗🤗🤗", duration=5)

    @torch.no_grad()
    def init_retargeting_image(self, retargeting_source_scale: float, source_eye_ratio: float, source_lip_ratio: float,
                               input_image=None):
        """ initialize the retargeting slider
        """
        if input_image != None:
            args_user = {'scale': retargeting_source_scale}
            self.args = update_args(self.args, args_user)
            self.cropper.update_config(self.args.__dict__)
            # inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source portrait ########
            img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=16)
            log(f"Load source image from {input_image}.")
            crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
            if crop_info is None:
                raise ValueError("Source portrait NO face detected", duration=2)
            source_eye_ratio = calc_eye_close_ratio(crop_info['lmk_crop'][None])
            source_lip_ratio = calc_lip_close_ratio(crop_info['lmk_crop'][None])
            self.source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
            self.source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)
            log("Calculating eyes-open and lip-open ratios successfully!")
            return self.source_eye_ratio, self.source_lip_ratio
        else:
            self.source_eye_ratio, self.source_lip_ratio = source_eye_ratio, source_lip_ratio
            return source_eye_ratio, source_lip_ratio
