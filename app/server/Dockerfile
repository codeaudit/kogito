FROM python:3.8

ENV HOME=/root
ENV APP_DIR=${HOME}/kogito/app/server
ENV FLASK_APP=server

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install dependencies
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends vim wget unzip

WORKDIR ${HOME}
ARG GITHUB_PERSONAL_TOKEN

# Clone kogito
RUN git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/epfl-nlp/kogito.git

WORKDIR ${APP_DIR}

# Setup app dependencies
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu114
RUN python3 -c "import nltk;nltk.download('punkt');nltk.download('wordnet');nltk.download('omw-1.4')"
RUN python3 -m spacy download en_core_web_sm

EXPOSE 8080

# gunicorn -w 1 -b 0.0.0.0:8080 --log-file server.log --capture-output --timeout 3600 --pid pid.log server:app --daemon
ENTRYPOINT ["flask", "run", "-h", "0.0.0.0", "-p", "8080"]