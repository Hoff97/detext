# Pull base image
FROM node:10.16

# Set work directory
WORKDIR /code

ENV PATH /code/node_modules/.bin:$PATH

# Copy dependencies
COPY package.json package-lock.json /code/

# Install dependencies
RUN npm install

# Copy project
COPY . /code/

# Build
RUN npm run build-prod

# Test
RUN ls ./dist/client

# Copy build
RUN mkdir -p  /usr/src/app/client
RUN cp -r ./dist/client/ /usr/src/app/

RUN ls /usr/src/app/client

CMD ["bash", "copy_build.sh"]
