# run mot demo
import mmcv
import tempfile
from collections import defaultdict
from mmtrack.apis import inference_mot, init_model
mot_config = './configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
input_video = './data/DNP/video/output000.mp4'
imgs = mmcv.VideoReader(input_video)
# build the model from a config file
mot_model = init_model(mot_config, device='cuda:0')
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
pred_file = "./output/log.json"
out_data = defaultdict(list)
# test and show/save the images
for i, img in enumerate(imgs):
        result = inference_mot(mot_model, img, frame_id=i)
        out_data[i].append(result)
        mot_model.show_result(
                img,
                result,
                show=False,
                wait_time=int(1000. / imgs.fps),
                out_file=f'{out_path}/{i:06d}.jpg')
        prog_bar.update()
mmcv.dump(out_data, pred_file)
output = './demo/mot.mp4'
print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()