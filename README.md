# Models & KWS Model Training Guide x86_64

### Testing your microphone

```
cd python_mic
python3 mic_test.py --list-devices
python3 mic_test.py --input-device 1 --output-device 3 -c 1 --samplerate 44100
```

### Testing KWS

```
cd python_mic
python3 mic_streaming.py --input-device 1 -k marvin -m ../trained_models/tflite/crnn_state_marvin.tflite
python3 mic_streaming.py --input-device 1 -k sheila -m ../trained_models/tflite/crnn_state_sheila.tflite
python3 mic_streaming.py --input-device 1 -k sheila,marvin -m ../trained_models/tflite/crnn_state_sheila_marvin.tflite
```

# Training guide

```
sudo apt update
sudo apt install wget python3-dev python3-pip python3-venv python3.7-distutils python3-pydot graphviz git apt-transport-https curl gnupg -y
```

### Prepare audio data

```
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir data
tar -xzf speech_commands_v0.02.tar.gz -C data
```

### Prepare KWS

```
cd ~
git clone https://github.com/overplex/kws.git
cd kws
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install numpy==1.19.5
pip3 install tensorflow==2.4.1
pip3 install tensorflow_model_optimization
pip3 install pydot
pip3 install graphviz
pip3 install absl-py
```

### Bazel

```
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update
sudo apt install bazel-3.1.0
```

### TensorFlow Addons v0.12.1

```
wget https://github.com/tensorflow/addons/archive/v0.12.1.zip
unzip v0.12.1.zip
cd addons-0.12.1
python3 configure.py
sudo ln -s ~/kws/venv/lib/python3.7/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so /usr/lib/lib_pywrap_tensorflow_internal.so
bazel-3.1.0 build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip install artifacts/tensorflow_addons-*.whl
```

Message about the mismatch of typing-extensions 4.7.1 version - OK.

### Training KWS

```
cd ~/kws
chmod +x train_crnn_state.sh
./train_crnn_state.sh
```

# Testing KWS

```
python3 mic_streaming.py --input-device 1 -k sheila,marvin -m ../models/crnn_state/tflite_stream_state_external/stream_state_external.tflite
```

# Tested on configuration

- Ubuntu 22.04.4 LTS
- Linux 5.10.102.1-microsoft-standard-WSL2 #1 SMP Wed Mar 2 00:30:59 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
- Python 3.7.17
<details>
  <summary>
    Python packages
  </summary>

```
blinker==1.4
cryptography==3.4.8
dbus-python==1.2.18
distro==1.7.0
distro-info==1.1+ubuntu0.2
graphviz==0.20.1
jeepney==0.7.1
launchpadlib==1.10.16
  - httplib2 [required: Any, installed: 0.20.2]
    - pyparsing [required: >=2.4.2,<4,!=3.0.3,!=3.0.2,!=3.0.1,!=3.0.0, installed: 2.4.7]
  - importlib-metadata [required: Any, installed: 4.6.4]
    - typing-extensions [required: >=3.6.4, installed: 4.7.1]
  - keyring [required: Any, installed: 23.5.0]
  - lazr.restfulclient [required: >=0.9.19, installed: 0.14.4]
    - httplib2 [required: >=0.7.7, installed: 0.20.2]
      - pyparsing [required: >=2.4.2,<4,!=3.0.3,!=3.0.2,!=3.0.1,!=3.0.0, installed: 2.4.7]
    - importlib-metadata [required: Any, installed: 4.6.4]
      - typing-extensions [required: >=3.6.4, installed: 4.7.1]
  - lazr.uri [required: Any, installed: 1.0.6]
  - six [required: Any, installed: 1.15.0]
more-itertools==8.10.0
pip==24.0
pipdeptree==2.9.6
pydot==1.4.2
PyGObject==3.42.1
  - pycairo [required: >=1.16.0, installed: ?]
PyJWT==2.3.0
python-apt==2.4.0+ubuntu3
SecretStorage==3.3.1
tensorflow==2.4.1
  - absl-py [required: ~=0.10, installed: 0.15.0]
    - six [required: Any, installed: 1.15.0]
  - astunparse [required: ~=1.6.3, installed: 1.6.3]
    - six [required: >=1.6.1,<2.0, installed: 1.15.0]
    - wheel [required: >=0.23.0,<1.0, installed: 0.37.1]
  - flatbuffers [required: ~=1.12.0, installed: 1.12]
  - gast [required: ==0.3.3, installed: 0.3.3]
  - google-pasta [required: ~=0.2, installed: 0.2.0]
    - six [required: Any, installed: 1.15.0]
  - grpcio [required: ~=1.32.0, installed: 1.32.0]
    - six [required: >=1.5.2, installed: 1.15.0]
  - h5py [required: ~=2.10.0, installed: 2.10.0]
    - numpy [required: >=1.7, installed: 1.19.5]
    - six [required: Any, installed: 1.15.0]
  - Keras-Preprocessing [required: ~=1.1.2, installed: 1.1.2]
    - numpy [required: >=1.9.1, installed: 1.19.5]
    - six [required: >=1.9.0, installed: 1.15.0]
  - numpy [required: ~=1.19.2, installed: 1.19.5]
  - opt-einsum [required: ~=3.3.0, installed: 3.3.0]
    - numpy [required: >=1.7, installed: 1.19.5]
  - protobuf [required: >=3.9.2, installed: 3.20.3]
  - six [required: ~=1.15.0, installed: 1.15.0]
  - tensorboard [required: ~=2.4, installed: 2.11.2]
    - absl-py [required: >=0.4, installed: 0.15.0]
      - six [required: Any, installed: 1.15.0]
    - google-auth [required: >=1.6.3,<3, installed: 2.30.0]
      - cachetools [required: >=2.0.0,<6.0, installed: 5.3.3]
      - pyasn1-modules [required: >=0.2.1, installed: 0.3.0]
        - pyasn1 [required: >=0.4.6,<0.6.0, installed: 0.5.1]
      - rsa [required: >=3.1.4,<5, installed: 4.9]
        - pyasn1 [required: >=0.1.3, installed: 0.5.1]
    - google-auth-oauthlib [required: >=0.4.1,<0.5, installed: 0.4.6]
      - google-auth [required: >=1.0.0, installed: 2.30.0]
        - cachetools [required: >=2.0.0,<6.0, installed: 5.3.3]
        - pyasn1-modules [required: >=0.2.1, installed: 0.3.0]
          - pyasn1 [required: >=0.4.6,<0.6.0, installed: 0.5.1]
        - rsa [required: >=3.1.4,<5, installed: 4.9]
          - pyasn1 [required: >=0.1.3, installed: 0.5.1]
      - requests-oauthlib [required: >=0.7.0, installed: 2.0.0]
        - oauthlib [required: >=3.0.0, installed: 3.2.0]
        - requests [required: >=2.0.0, installed: 2.31.0]
          - certifi [required: >=2017.4.17, installed: 2024.6.2]
          - charset-normalizer [required: >=2,<4, installed: 3.3.2]
          - idna [required: >=2.5,<4, installed: 3.7]
          - urllib3 [required: >=1.21.1,<3, installed: 2.0.7]
    - grpcio [required: >=1.24.3, installed: 1.32.0]
      - six [required: >=1.5.2, installed: 1.15.0]
    - Markdown [required: >=2.6.8, installed: 3.4.4]
      - importlib-metadata [required: >=4.4, installed: 4.6.4]
        - typing-extensions [required: >=3.6.4, installed: 4.7.1]
    - numpy [required: >=1.12.0, installed: 1.19.5]
    - protobuf [required: >=3.9.2,<4, installed: 3.20.3]
    - requests [required: >=2.21.0,<3, installed: 2.31.0]
      - certifi [required: >=2017.4.17, installed: 2024.6.2]
      - charset-normalizer [required: >=2,<4, installed: 3.3.2]
      - idna [required: >=2.5,<4, installed: 3.7]
      - urllib3 [required: >=1.21.1,<3, installed: 2.0.7]
    - setuptools [required: >=41.0.0, installed: 59.6.0]
    - tensorboard-data-server [required: >=0.6.0,<0.7.0, installed: 0.6.1]
    - tensorboard-plugin-wit [required: >=1.6.0, installed: 1.8.1]
    - Werkzeug [required: >=1.0.1, installed: 2.2.3]
      - MarkupSafe [required: >=2.1.1, installed: 2.1.5]
    - wheel [required: >=0.26, installed: 0.37.1]
  - tensorflow-estimator [required: >=2.4.0,<2.5.0, installed: 2.4.0]
  - termcolor [required: ~=1.1.0, installed: 1.1.0]
  - typing-extensions [required: ~=3.7.4, installed: 4.7.1]
  - wheel [required: ~=0.35, installed: 0.37.1]
  - wrapt [required: ~=1.12.1, installed: 1.12.1]
tensorflow-addons==0.12.1
  - typeguard [required: >=2.7, installed: 4.1.2]
    - importlib-metadata [required: >=3.6, installed: 4.6.4]
      - typing-extensions [required: >=3.6.4, installed: 4.7.1]
    - typing-extensions [required: >=4.7.0, installed: 4.7.1]
tensorflow-model-optimization==0.7.3
  - dm-tree [required: ~=0.1.1, installed: 0.1.8]
  - numpy [required: ~=1.14, installed: 1.19.5]
  - six [required: ~=1.10, installed: 1.15.0]
unattended-upgrades==0.1
wadllib==1.3.6
  - importlib-metadata [required: Any, installed: 4.6.4]
    - typing-extensions [required: >=3.6.4, installed: 4.7.1]
zipp==1.0.0
```
</details>

# Acknowledgements

- [@petewarden](https://github.com/petewarden) for [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209)
- [@rybakov](https://github.com/rybakov) for [Streaming Aware neural network models](https://github.com/google-research/google-research/tree/master/kws_streaming)
- [@StuartIanNaylor](https://github.com/StuartIanNaylor) for [Dataset-builder](https://github.com/StuartIanNaylor/Dataset-builder)
- [@StuartIanNaylor](https://github.com/StuartIanNaylor) for [google-kws](https://github.com/StuartIanNaylor/g-kws)
- [@SaneBow](https://github.com/SaneBow) for [TFLiteKWS](https://github.com/SaneBow/tflite-kws/)