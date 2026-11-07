set -e

docker build --platform linux/amd64 -t noscribewitheditor-cuda-pyinstaller . 2> output/build.log

docker run \
--platform linux/amd64 \
-v "${PWD}/spec_files:/spec_files:ro" \
-v "${PWD}/desktop_files:/desktop_files:ro" \
-v "${PWD}/../../..:/noScribe:ro" \
-v "${PWD}/../noScribeEditor:/noScribeEditor:ro" \
-v "${PWD}/output:/output" \
noscribewitheditor-cuda-pyinstaller &> output/pyinstaller_noScribe_linux.log
