echo Downloading pretrained model...
mkdir model
wget https://drive.google.com/file/d/1bQvDgCRIdJWV5vi0iuZ5ymSIjxdNU3Vx/view?usp=drive_link
mv network-snapshot_000320.pkl ./model

echo Downloading preprocessed smpl sdf...
wget wget https://polybox.ethz.ch/index.php/s/Q5pyLvX4ECXEktR/download
mv ./download ./model/sdf_smpl.npy

echo Downloading pose distribution of deepfashion...
mkdir data
wget https://polybox.ethz.ch/index.php/s/mbIsbTVrktIqPJm/download
mv ./download ./data/dp_pose_dist.npy

