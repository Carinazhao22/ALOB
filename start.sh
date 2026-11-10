nohup python -u train.py --cfg /home/student/Desktop/Carina/TwoNets/experiments/CASIA-112x96-LMDB-Mask.yaml \
                    --model 'LResNet50E_IR_FPN'\
                    --batch_size 128 \
                    --gpus '0' \
                    --lr 0.01 \
                    --weight_pred 0.5 \
                    --optim 'sgd'\
                    --ratio 3 \
                    --pattern 5 \
                    --debug 0
                    >>nohup.out 2>&1 &
