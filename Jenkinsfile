node("rocmtest") {
    docker.image('petrex/rocm_caffe2')
    withDockerContainer(image: "petrex/rocm_caffe2", args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
        timeout(time: 2, unit: 'HOURS'){
            // sh 'groups'
            stage("checkout") {
                checkout scm
                sh 'git submodule update --init'
            }
            
            //stage('Clang Format') {
              //  sh '''
                //    find . -iname *miopen* -o -iname *hip* \
                  //  | grep -v 'build/' \
                    //| xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-3.8-style=file {} | diff - {}'
                //'''
            //}

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
        }
    }
}

/*
def rocmtestnode(variant, name, body) {
    def image = 'miopen'
    def cmake_build = { compiler, flags ->
        def cmd = """
            echo \$HSA_ENABLE_SDMA
            mkdir -p $WINEPREFIX
            rm -rf build
            mkdir build
            cd build
            CXX=${compiler} CXXFLAGS='-Werror' cmake -DMIOPEN_GPU_SYNC=On -DCMAKE_CXX_FLAGS_DEBUG='-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined' ${flags} .. 
            CTEST_PARALLEL_LEVEL=4 dumb-init make -j32 check doc MIOpenDriver
        """
        echo cmd
        sh cmd
    }
    node(name) {
        stage("checkout ${variant}") {
            // env.HCC_SERIALIZE_KERNEL=3
            // env.HCC_SERIALIZE_COPY=3
            env.HSA_ENABLE_SDMA=0 
            // env.HSA_ENABLE_INTERRUPT=0
            env.WINEPREFIX="/jenkins/.wine"
            checkout scm
            sh 'git submodule update --init'
        }
        
        //stage('Clang Format') {
          //  sh '''
            //    find . -iname *miopen* -o -iname *hip* \
              //  | grep -v 'build/' \
                //| xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-3.8-style=file {} | diff - {}'
            //'''
        //}

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
*/
