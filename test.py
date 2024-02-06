import cv2
from PIL import Image
import numpy as np
from termcolor import colored
from vima_utils import any_slice, get_batch_size
from vima_bench import make
from vima_bench.tasks.utils.misc_utils import obs2bbox
import warnings

def draw_bbox(img, bbox, color=(0, 0, 255), thickness=2):
    xmin, ymin, xmax, ymax = bbox
    # img is pil image
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    return img

def plot_rgb(rgb, file_name):
    plot_rgb = rgb.transpose(1, 2, 0)
    plot_rgb = np.array(plot_rgb, dtype=np.uint8)
    # draw bbox
    plot_rgb = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR) # cv2 uses BGR
    # for bbox in bbox_dict[view].values():
    #     plot_rgb = draw_bbox(plot_rgb, bbox)
    plot_rgb = Image.fromarray(plot_rgb[:, :, ::-1]) # convert back to RGB and PIL image
    plot_rgb.save(file_name)

def rearrange(arr, pattern):
    if isinstance(pattern, str):
        pattern = pattern.split(" ")
    return np.transpose(arr, [pattern.index(x) for x in "hwc"])

def plot_action_on_rgb(rgb, action, action_space):
   _, h, w = rgb.shape
   pos0, pos1 = action["pose0_position"], action["pose1_position"]
   # normalize to [0, 1] then scale to image size
   pos0 = (pos0 - action_space['low']) / (
       action_space['high'] - action_space['low']
   )
   pos1 = (pos1 - action_space['low']) / (
       action_space['high'] - action_space['low']
   )
   pos0 = pos0 * np.array([h, w])
   pos1 = pos1 * np.array([h, w])
   print(pos0, pos1)
   # annotate rgb
   rgb = rearrange(rgb.copy(), "c h w -> h w c")
   # RGB -> BGR
   rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
   # draw circles
   rgb = cv2.circle(rgb, tuple(pos0.astype(np.int32)[::-1]), 5, (0, 0, 255), 2)
   rgb = cv2.circle(rgb, tuple(pos1.astype(np.int32)[::-1]), 5, (0, 255, 0), 2)
   # put text
   rgb = cv2.putText(
       rgb,
       " pick",
       org=tuple(pos0.astype(np.int32)[::-1]),
       fontScale=0.5,
       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
       color=(0, 0, 255),
       thickness=1,
       lineType=cv2.LINE_AA,
   )
   rgb = cv2.putText(
       rgb,
       " place",
       org=tuple(pos1.astype(np.int32)[::-1]),
       fontScale=0.5,
       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
       color=(0, 255, 0),
       thickness=1,
       lineType=cv2.LINE_AA,
   )
   # BGR -> RGB
   rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
   return rgb

def get_action(bbox_dict, object_ids, action_bounds, h, w):
    action = {}
    center_object_0 = (bbox_dict[object_ids[0]][:2] + bbox_dict[object_ids[0]][2:]) / 2
    center_object_1 = (bbox_dict[object_ids[1]][:2] + bbox_dict[object_ids[1]][2:]) / 2
    # flip x and y
    center_object_0 = center_object_0[::-1]
    center_object_1 = center_object_1[::-1]
    # normalize with height and width
    center_object_0 = center_object_0 / np.array([h, w])
    center_object_1 = center_object_1 / np.array([h, w])
    center_object_0 = center_object_0 * (action_bounds["high"] - action_bounds["low"]) + action_bounds["low"]
    center_object_1 = center_object_1 * (action_bounds["high"] - action_bounds["low"]) + action_bounds["low"]
    action["pose0_position"] = center_object_0
    action["pose1_position"] = center_object_1
    action["pose0_rotation"] = np.asarray([0, 0, 0, 1])
    action["pose1_rotation"] = np.asarray([0, 0, 0, 1])
    return action

def main():
    view = "top"
    task_kwargs = {
            "dragged_obj_express_types": "name",
            "base_obj_express_types": "name",
    }
    env = make(
            task_name="visual_manipulation",
            hide_arm_rgb=True,
            task_kwargs=task_kwargs,)
    env.seed(100)
    obs = env.reset()
    prompt, prompt_assets, meta_info = env.prompt, env.prompt_assets, env.meta_info
    action_bounds = meta_info["action_bounds"]
    print(meta_info["seed"])
    print(prompt)
    prompt2obj_id = {}
    for name, placeholder in env.task.placeholders.items():
        prompt2obj_id[name] = placeholder.obj_id
    print(prompt2obj_id)
    object_ids = list(prompt2obj_id.values())[::-1]
    objects = list(meta_info["obj_id_to_info"].keys())
    # print meta_info in green
    print(colored(meta_info.keys(), "green"))
    print(colored(action_bounds, "green"))

    rgb_dict, bbox_dict = obs2bbox(obs, list(meta_info["obj_id_to_info"].keys()))
    rgb = rgb_dict[view]
    _, h, w = rgb.shape
    plot_rgb(rgb, "rgb.png")


    action = get_action(bbox_dict[view], object_ids, action_bounds, h, w)

    img_w_action = plot_action_on_rgb(rgb, action, meta_info["action_bounds"]) # returns in RGB format, good for PIL image saving
    img_w_action = Image.fromarray(img_w_action)
    img_w_action.save("rgb_w_action.png")

    obs, reward, done, info = env.step(action)

    rgb = obs.get("rgb")[view]
    plot_rgb(rgb, "rgb_a.png")



if __name__ == '__main__':
    main()
