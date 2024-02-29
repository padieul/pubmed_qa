# Use an official Node.js runtime as a base image
FROM node:14

# Set the working directory to /app/frontend
WORKDIR /app/frontend

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code to the working directory
COPY . .

# Expose port 3000 (assuming your Node.js application is running on this port)
EXPOSE 3000

# Command to run the application
CMD ["node", "index.js"]
