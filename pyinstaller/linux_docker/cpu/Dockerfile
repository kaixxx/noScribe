FROM python:3.12.10-bullseye

WORKDIR /output

ADD requirements/requirements_linux_nS.txt .

ADD requirements/requirements_linux_nSE.txt .

SHELL ["/bin/bash", "-c"]

RUN python -m venv /venv/nS && source /venv/nS/bin/activate && pip install torchaudio==2.8 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements_linux_nS.txt && deactivate

RUN python -m venv /venv/nSE && source /venv/nSE/bin/activate && pip install -r requirements_linux_nSE.txt && deactivate

CMD ["/bin/bash", "-c", "source /venv/nS/bin/activate && pip freeze && pyinstaller --clean --noconfirm /spec_files/noScribe.spec && deactivate && mkdir to_dist && cp /desktop_files/noScribe.desktop to_dist/noScribe.desktop && mv dist/noScribe to_dist/noScribe && rm -rf build noScribe.spec dist && source /venv/nSE/bin/activate && pip freeze && pyinstaller --noconfirm --onedir --windowed --icon /noScribeEditor/noScribeEditLogo.ico --add-data /noScribeEditor/noScribeEditLogo.png:. /noScribeEditor/noScribeEdit.py && deactivate && cp /desktop_files/noScribeEdit.desktop to_dist/noScribeEdit.desktop && mv dist/noScribeEdit to_dist/noScribe/_internal/noScribeEdit && rm -rf build noScribeEdit.spec dist && cd to_dist && tar -zcvf noScribe_cpu_linux_amd64.tar.gz noScribe noScribe.desktop noScribeEdit.desktop"]
