# Regular
python ./eval_nus.py --video_dir=Regular/original/videos --motion_dir=Regular/original/motions/mesh --model_path=./pretrained/pretrained_model.pth.tar
# QuickRotation
python ./eval_nus.py --video_dir=QuickRotation/original/videos --motion_dir=QuickRotation/original/motions/mesh --model_path=./pretrained/pretrained_model.pth.tar
# Zooming
python ./eval_nus.py --video_dir=Zooming/original/videos --motion_dir=Zooming/original/motions/mesh --model_path=./pretrained/pretrained_model.pth.tar
# Crowd
python ./eval_nus.py --video_dir=Crowd/original/videos --motion_dir=Crowd/original/motions/mesh --model_path=./pretrained/pretrained_model.pth.tar
# Parallax
python ./eval_nus.py --video_dir=Parallax/original/videos --motion_dir=Parallax/original/motions/mesh --model_path=./pretrained/pretrained_model.pth.tar
# Running
python ./eval_nus.py --video_dir=Running/original/videos --motion_dir=Running/original/motions/mesh --model_path=./pretrained/pretrained_model.pth.tar
