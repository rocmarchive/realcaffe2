#!/bin/bash
total_test_count=0
passed_count=0
failed_tests=()
ignore_tests=()
for T in $(ls); do
    if [[ "$T" =~ "test" ]]; then
	if [[ "${ignore_tests[*]}" =~ "$T" ]]; then
	    echo "+++++++++++++++++++ ignored ++++++++++++++++++++++"
            echo "$T"
	    continue
	fi
	total_test_count=$((total_test_count+1))
        echo $T
        ./$T
        if [ $? -eq 0 ]; then
    	    passed_count=$((passed_count+1))    
        else
	    failed_tests+=($T)
	fi
    fi
done

echo "passed_count:"
echo $passed_count
echo "total_count:"
echo $total_test_count

if [ $passed_count -eq $total_test_count ]; then
    echo "All passed"
    exit 0
else
    echo "Failed tests..."
    echo ${failed_tests[*]}
    exit 1
fi

