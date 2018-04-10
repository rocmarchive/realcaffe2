#!/bin/bash
total_test_count=$(ls | wc -l)	
echo $total_test_count
passed_count=0
failed_tests=()
for T in $(ls); do
    echo $T
    ./$T
    if [ $? -eq 0 ]; then
    	passed_count=$((passed_count+1))    
    else
        failed_tests+=($T)
	fi
    done
    if [ $passed_count -eq $total_test_count ]; then
        echo "All passed"
        exit 0
    else
        echo "Failed tests..."
        echo ${failed_tests[*]}
	echo "passed_count:"
	echo $passed_count
	echo "total_count:"
	echo $total_test_count
        exit 1
    fi
echo "done"
