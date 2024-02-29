Main experiments for our paper ["InferDB: In-Database Machine Learning Inference Using Indexes"](https://hpi.de/fileadmin/user_upload/fachgebiete/rabl/publications/2024/inferdb_vldb24.pdf).

We evaluate InferDB on a server with an AMD-EPYC-7742 2.25GHz CPU (10 cores used), 80 GiB RAM, and 8x3.2 TB SAS SSDs (RAID5).

# Requisites

- Docker

# Supported CPU architechtures:

- arm64
- amd64

# Running the experiments

First, clone the code repository and download InferDB's docker image:

```bash

git clone https://github.com/hpides/inferdb

docker pull hpides/inferdb

```

Start a container. The entryscript of the image will configure the Postgres instance, install the PGML extension, and uncompress the data used in the paper's main experiments:

```bash

docker run -it --shm-size=4g --memory-swap -1 --name inferdb hpides/inferdb sleep infinity

```

To reproduce the standalone experiments, Table 1 and Table 2 in the paper, run:

```bash

# --iterations control the number of runs per model for each dataset
# --paper_data if true only considers the models reported in table 2
# To approximate the reported results you should go with: --iterations 5 --paper_models True
# To get all results set: --iterations 5 --paper_models False

docker exec -it inferdb bash /app/inferdb/bash_scripts/standalone_experiments/standalone_experiments.sh --iterations 1 --paper_models True

```

To reproduce the postgres experiments, Table 3 in the paper, run:

```bash

# --iterations control the number of runs per model for each dataset
# --paper_data if true only considers the models reported in table 2
# To approximate the reported results you should go with: --iterations 5 --paper_models True
# To get all results set: --iterations 5 --paper_models False

docker exec -it inferdb bash /app/inferdb/bash_scripts/pg_experiments/pg_experiments.sh --iterations 1 --paper_models True

```

Copy the tables from the container to the host:

```bash

docker cp inferdb:/app/inferdb/experiments/plots/latex/output/ ./inferdb/experiments/plots/latex/

```

Docker image was built using:

```bash

docker buildx build -f Dockerfile.local --platform linux/amd64,linux/arm64 -t hpides/inferdb .

```