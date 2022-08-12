FROM node:18.5

ENV HOME=/root
ENV APP_DIR=${HOME}/kogito-client
ENV REACT_APP_SERVER_URL=http://localhost:8080

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install dependencies
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends vim wget unzip

RUN mkdir ${APP_DIR}
WORKDIR ${APP_DIR}

# Copy server files
COPY . .

# Setup app dependencies
RUN npm install -g serve
RUN npm install
RUN npm run build

EXPOSE 3000

CMD ["serve", "-s", "build"]