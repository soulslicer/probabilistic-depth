sudo apt-get install libpng++-dev libpcl-dev libproj-dev libopencv-dev
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0  /usr/lib/x86_64-linux-gnu/libEGL.so
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libGL.so.1.7.0  /usr/lib/x86_64-linux-gnu/libGL.so
sudo ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so
cd deval_lib
rm -rf build; mkdir build; cd build;
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cp pyevaluatedepth_lib.cpython-35m-x86_64-linux-gnu.so ../pyevaluatedepth_lib.so
cd ..; cd ..
cd perception_lib
rm -rf build; mkdir build; cd build;
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cp pyperception_lib.cpython-35m-x86_64-linux-gnu.so ../pyperception_lib.so
cp libperception_lib.so ../
cd ..; cd ..
cd utils_lib
rm -rf build; mkdir build; cd build;
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cp utils_lib.cpython-35m-x86_64-linux-gnu.so ../utils_lib.so
cd ..; cd ..
