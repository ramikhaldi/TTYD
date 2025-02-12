# Use the latest Node.js image
FROM node:latest

WORKDIR /app

# Copy package files first for efficient caching
COPY package.json yarn.lock ./

# Install dependencies, including react-markdown
RUN yarn install

# Copy all files after installing dependencies
COPY . ./

# Start the React application
CMD ["yarn", "start"]
