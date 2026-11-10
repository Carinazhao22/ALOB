python -u test.py --cfg experiments/CASIA-112x96-LMDB-Mask.yaml \
                  --model 'LResNet50E_IR_FPN'\
                  --batch_size 128 \
                  --gpus '0' \
                  --ratio 3 \
                  --pattern 5 \
                  --debug 1
 
# ar face dataset testing
# python -u lib/core/ar_dataset.py --protocol 2

# root_path='/content/drive/MyDrive/FROM/data/datasets/ARdata/images_112_96/gallery_masked'
# python -u lib/core/ar_dataset.py --glass-root $root_path --protocol 2

# generate npy files for megaface testing
# path='/content/drive/MyDrive/datasets/facescrub_images_masked'
# path='/content/drive/MyDrive/datasets/facescrub_images'
# python -u lib/core/megaface_mp.py --facescrub-root $path\
#                                   --megaface-flag 1 \
#                                   --gpus '0' \
#                                    -j 1
