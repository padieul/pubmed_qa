
## Installation Instructions 

Please follow the the steps below to setup the infrastructure for the RAG project

1. Install the latest version of [`Docker Desktop`](https://www.docker.com/products/docker-desktop/)
2. Install the required libraries to run the Python script [`data_to_os.py`](data_preprocessing/data_to_os.py) that upload the data to [`OpenSearch`](https://opensearch.org/), the required libraries can be found in [`requirements.txt`](requirements.txt)

    pip install -r requirements.txt

3. Create a folder in [`data_preprocessing`](data_preprocessing) and name it `data` 

4. Copy the data file [`data_embeddings_500_100.csv`](https://1drv.ms/f/s!AgO6RudpYGaRg7k2pyaJ62r_GukU7g?e=aIeKbj) in OneDrive to the folder [`data_preprocessing/data/`](data_preprocessing/data/)

    > The data file is about 5 GB!

5. Run the Python script [`data_to_os.py`](data_preprocessing/data_to_os.py) to upload the data to [`OpenSearch`](https://opensearch.org/)

    > This step might take 15 to 20 minutes, depending on the speed of the computer.
    > Please make sure the computer is not under heavy load during this step and if it fails please rerun the script again.
    > After a minute, it is expected to see a progress bar as the one below.

    <div style="text-align:center"><img src="images/progress.png" /></div>

6. Open the front-end main webpage [`http://localhost:3000/`](http://localhost:3000/) and wait for the RAG pipeline to be fully initialized

    <div style="text-align:center"><img src="images/backend_init.png" /></div>

7. Once the initialization is done, you will see that the server is ready

    <div style="text-align:center"><img src="images/server_ready.png" /></div>

8. Now you can ask the question you want

    <div style="text-align:center"><img src="images/question.png" /></div>