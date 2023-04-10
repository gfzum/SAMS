#/bin/bash
cpu=0
name='cws_check'
checkpoint=gofiles/checkpoints/pku1/pku1_model_epoch_best_step_0.pt
testfile=testing/pku_test.utf8
# testfile=gofiles/logs/pku1.log
type=pku

gold=icwb2-data/gold/pku_test_gold.utf8
words=icwb2-data/gold/pku_training_words.utf8

batch_size=16
showsteps=100

log=${testfile}.log

bash pro.sh $testfile $type

savefile=${testfile}_eval

        # --gpu \
# CUDA_VISIBLE_DEVICES=$cpu   
python valid_single.py --logfiles $log \
        --name $name \
        --evalfile  ${testfile}_repo \
        --savefiles $savefile \
        --showsteps $showsteps \
        --valid_batch_size $batch_size \
        --use_buffers \
        --model  $checkpoint



bash repo.sh $testfile $savefile ${savefile}_repo $type



python calculatePRF1.py --pred ${savefile}_repo --gold ${gold} --word ${words}
