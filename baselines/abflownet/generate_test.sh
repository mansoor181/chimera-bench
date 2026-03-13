#!/bin/bash

TEST_SCRIPT="design_testset.py"

for index in {0..57}; do
    echo "Running test for index: $index"
    export CUDA_VISIBLE_DEVICES=4

    python "$TEST_SCRIPT" $index --config "configs/test/codesign_single.yml"

    if [ $? -eq 0 ]; then
        echo "Test for index $index completed successfully."
    else
        echo "Test for index $index failed."
    fi

    echo "----------------------------------"
done

echo "All tests completed."