node("rocmtest14") {
    sh ''' docker login --username rohith612 --password 123456 '''
    docker.image('rohith612/rocm_caffe2:clang-format')
    withDockerContainer(image: "rohith612/rocm_caffe2:clang-format", args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
        timeout(time: 2, unit: 'HOURS'){
            stage("checkout") {
                checkout scm
                sh 'git submodule update --init'
            }
            
            stage('clang_format') {
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
                    cd build/bin
                    ../../test.sh
                '''
            }
            stage("inference_test"){
                sh '''
                export PYTHONPATH=$PYTHONPATH:~/build
                export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
                model=resnet50
                python caffe2/python/models/download.py $model    
                cd build/bin
                python ../../tests/inference_test.py -m $model -s 224 -e 1
                '''

            }
        }
    }
}
