#!/bin/bash
if [ -f "./aug_colorfinetune_169.tar" ]; then
    echo "aug_colorfinetune_169.tarexists, skip download."
else
    wget https://www.dropbox.com/s/87opqlfs4b95e6d/aug_colorfinetune_169.tar
    
fi
if [ -f "./aug_resolutionfinetune_146.tar" ]; then
    # 檔案 /path/to/dir/filename 存在
    echo "aug_resolutionfinetune_146.tar exists, skip download."
else
    wget https://www.dropbox.com/s/y5k65bv3gkdqrvd/aug_resolutionfinetune_146.tar
fi

if [ -f "./synthesis_tune_best_for_end.tar" ]; then
    # 檔案 /path/to/dir/filename 存在
    echo "synthesis_tune_best_for_end.tar exists, skip download."
else
    wget https://www.dropbox.com/s/7mppenwgz1eyxrc/synthesis_tune_best_for_end.tar
fi
