import os
import cv2
import numpy as np
from vima_utils import any_slice, get_batch_size
from vima_bench import make
from vima_bench import ALL_PARTITIONS, PARTITION_TO_SPECS
from vima_bench.tasks.utils.misc_utils import get_dragged_obj_bbox, draw_bbox

def main():
    view = "top"
    task_kwargs = PARTITION_TO_SPECS["train"]["visual_manipulation"]
    env = make(
            task_name="visual_manipulation",
            hide_arm_rgb=True,
            task_kwargs=task_kwargs,)
    env.seed(0)
    env.reset()
    rgb_list = []
    for _ in range(50):
        obs = env.reset()
        prompt2obj_id = {}
        object_id = env.task.placeholders['dragged_obj_1'].obj_id
        prompt, prompt_asset = env.get_prompt_and_assets()

        rgb = obs.get("rgb")[view]
        bbox = get_dragged_obj_bbox(obs.get('segm'), view, object_id)
        rgb = draw_bbox(rgb, bbox)
        rgb_list.append(rgb)
    # convert to one big image
    rgb = np.concatenate(rgb_list, axis=1)
    # save to file
    filename = "test_object_loc.png"
    rgb = rgb.astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))  # this is rgb
    print(rgb.shape)

    from PIL import Image
    img = Image.fromarray(rgb)
    img.save(filename)


if __name__ == '__main__':
    main()
