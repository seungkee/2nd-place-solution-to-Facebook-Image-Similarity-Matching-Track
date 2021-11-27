find . -type f -iname "*.sh" -exec chmod +x {} \;
cd /submission/code
./install.sh
cd /submission
./makesoftlink.sh
cd /submission/code/descriptor_track/train_eval/
./descriptor_run_debug.sh
cd /submission/code/matching_track/train_eval/
./matching_run.sh
