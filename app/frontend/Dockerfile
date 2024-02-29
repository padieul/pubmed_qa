# Use an official Node.js runtime as the base image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /usr/src/app

# Optionally set NODE_ENV to development to ensure devDependencies are installed
ENV NODE_ENV=development

# Copy package.json and package-lock.json to the working directory
# Run install rollup and sirv-cli explicitly (although actually it should be installed as devDependencies from package.json)
# TODO: investigate why rollup and sirv-cli are not installed implicitly
RUN npm install -g rollup
RUN npm install -g sirv-cli
COPY package*.json ./

# Install dependencies including devDependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the Svelte app
RUN npm run build

# Optionally, set NODE_ENV back to production (or remove if not necessary)
ENV NODE_ENV=production

# Expose the port that the app runs on
EXPOSE 3000

# Command to run the application
CMD ["npm", "run", "start"]

