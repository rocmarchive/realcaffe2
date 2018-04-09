node("rocmtest") {
    docker.image('rohith612/rocm_caffe2:clang-format')
    withDockerContainer(image: "rohith612/rocm_caffe2:clang-format", args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
        timeout(time: 2, unit: 'HOURS'){
            stage("checkout") {
                checkout scm
                sh 'git submodule update --init'
            }
            
            stage('Clang Format') {
                sh '''
                    cd caffe2
                    find . -iname *miopen* -o -iname *hip* \
                    | grep -v 'build/' \
                    | xargs -n 1 -P 1 -I{} -t sh -c 'clang-format-3.8 -style=file {} | diff - {}'
                '''
            }
            stage("build_release") {

                sh '''
                    export HCC_AMDGPU_TARGET=gfx900
                    rm -rf build
                    mkdir build
                    cd build
                    cmake -DCMAKE_BUILD_TYPE='Release' ..
                    make -j8
                    make DESTDIR=./install install
                '''
            }
            /*
            stage("build_debug") {
                sh '''
                    rm -rf build
                    mkdir build
                    cd build
                    cmake -DCMAKE_BUILD_TYPE='Debug' ..
                    make -j8
                    make DESTDIR=./install install
                '''
            }
            */
            stage("binary_tests") {
                sh '''
                    //set -e
                    cd build/bin
                    total_test_count=$(ls | wc -l)
                    echo $total_test_count
                    //passed_tests=()
                    passed_count=0
                    failed_tests=()
                    for T in $(ls); do
                        echo $T
                        ./$T
                        if [ $? -eq 0 ]; then
                            passed_count=$((passed_count+1))
                            //passed_tests+=($T)
                        else
                            failed_tests+=($T)
                        fi

                    done
                    if [ $passed_count -eq $total_test_count ]; then
                        echo "All passed"
                        exit 0
                    else
                        echo "Failed tests..."
                        echo ${failed_test[*]}
                        exit 1
                    fi
                    echo "done"
                '''
            }
        }
    }
}
