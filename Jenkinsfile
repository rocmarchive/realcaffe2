node("rocmtest") {
    stage("checkout") {
        checkout scm
    }
    stage("docker_pull") {
        sh '''
             docker login -u rohith612 -p 123456
             docker pull petrex/rocm_caffe2
           '''

    }
    withDockerContainer(image: "rocm_caffe2", args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
        timeout(time: 1, unit: 'HOURS') {
            stage('Clang Format') {
                sh '''
                    find . -iname *miopen* -o -iname *hip* \
                    | grep -v 'build/' \
                    | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-3.8-style=file {} | diff - {}'
                '''
            }
            stage("build_debug") {
                sh '''
                    rm -rf build
                    mkdir build
                    cd build
                    cmake -DCMAKE_BUILD_TYPE='Debug' ..
                    make -j8
                    make install
                '''
            }
            stage("build_release") {
                sh '''
                    rm -rf build
                    mkdir build
                    cd build
                    cmake -DCMAKE_BUILD_TYPE='Release' ..
                    make -j8
                    make install
                '''
            }

        }
    }
}

