node("rocmtest14") {
    
    sh 'docker login --username rohith612 --password 123456'
    stage("checkout") {
        checkout scm
        sh 'git submodule update --init'
    }
    
    stage("docker_image") {
        sh 'echo $PWD'
        sh './docker/ubuntu-16.04-rocm171/docker-build.sh rocm171'
    }
    
    withDockerContainer(image: "caffe2-rocm171", args: '--device=/dev/kfd --device=/dev/dri --group-add video -v $PWD:/rocm-caffe2') {
        timeout(time: 2, unit: 'HOURS'){
            stage('clang_format') {
                sh '''
                    cd caffe2
                    find . -iname *miopen* -o -iname *hip* \
                    | grep -v 'build/' \
                    | xargs -n 1 -P 1 -I{} -t sh -c 'clang-format-3.8 -style=file {} | diff - {}'
                '''
            }
            /*
            stage("build_debug") {

                sh '''
                    export HCC_AMDGPU_TARGET=gfx900
                    ls /data/Thrust
                    export THRUST_ROOT=/data/Thrust
                    echo $THRUST_ROOT
                    rm -rf build
                    mkdir build
                    cd build
                    cmake -DCMAKE_BUILD_TYPE='Debug' ..
                    make -j$(nproc)
                    make DESTDIR=./install install
                '''
            }
            */
            stage("build_release") {
                sh '''
                    export HCC_AMDGPU_TARGET=gfx900
                    export THRUST_ROOT=/data/Thrust
                    echo $THRUST_ROOT
                    rm -rf build
                    mkdir build
                    cd build
                    cmake -DCMAKE_BUILD_TYPE='Release' ..
                    make -j$(nproc)
                    make DESTDIR=./install install
                '''
            }
            stage("binary_tests") {
                sh '''
                    export LD_LIBRARY_PATH=/usr/local/lib
                    cd build/bin
                    ../../tests/test.sh
                '''
            }

            
            stage("python_op_tests"){
                sh '''
                    export LD_LIBRARY_PATH=/usr/local/lib
                    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
                    cd build/bin
                    ../../tests/python_tests.sh
                '''

            }
            
            
            stage("inference_test"){
                sh '''
                export PYTHONPATH=$PYTHONPATH:$(pwd)/build
                export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
                echo $PYTHONPATH
                model=resnet50
                python caffe2/python/models/download.py $model
                cd build/bin
                python ../../tests/inference_test.py -m ../../$model -s 224 -e 1
                '''

            }
            
        }
    }
}
