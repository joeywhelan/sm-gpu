![cover image](cover600x322.png)

# Elastic/Nvidia GPU Integration
This article is covers integration of Elastic index and embedding operations with Nvidia GPUs.  I create an Elastic cluster in Kubernetes via Elastic Cloud on Kubernetes (ECK) on a Google Kubernetes Engine (GKE) with Nvidia GPU-enabled nodes.  Topics covered:

- Provisiong of a [GKE](https://cloud.google.com/kubernetes-engine?hl=en) cluster with CPU-only and CPU+GPU nodes.
- Provisioning of an [ECK](https://www.elastic.co/docs/deploy-manage/deploy/cloud-on-k8s) cluster on GKE 
- Configuration of Elastic data nodes with Nvidia [CUDA](https://developer.nvidia.com/cuda/toolkit) and [cuVS](https://developer.nvidia.com/cuvs) libraries
- Deployment of the Elastic [GPU plugin](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/gpu-vector-indexing) providing GPU-accelerated HNSW indexing
- Creation of a synthetic data set with embeddings generated from a GPU-accelerated Jina.ai model hosted on [Elastic Inference Service](https://www.elastic.co/docs/explore-analyze/elastic-inference/eis) (EIS)
- Speed/throughput tests comparing CPU-only vs GPU-accelerated indexing

# Overall Architecture
![high-level architecture](highlevel.png) 

ECK is deployed on a GKE k8s architecture.  Nvidia GPU-enabled nodes are deployed in that GKE cluster. A Jina.ai embedding model ([jina-embeddings-v3](https://www.elastic.co/docs/explore-analyze/elastic-inference/eis#jina-embeddings-on-eis)) is leveraged on EIS.
## Low-level Architecture
![low-level architecture](lowlevel.png)

GKE nodes are deployed in three zones in 1 region (us-central1).  Two different node types are utilized - CPU-only and CPU+GPU.  CPU-only nodes are used house the Elastic Master and Kibana processes.  Elastic data nodes are put on the GPU-enabled nodes.

# Provisioning
## GKE Nodes
I use the [gcloud CLI](https://cloud.google.com/cli?hl=en) for provisioning GKE.  The commands below create 3 CPU-only and 3 GPU+CPU nodes in 3 zones - 1 per zone - in 1 GCP region.  Nvida L4 GPUs are used here, 1 per node.  Of note, I set this for 'spot' allocation to reduce the cost of instantiating these GPUs.

```bash
gcloud container clusters create gpu-demo \
    --region us-central1 \
    --node-locations us-central1-a,us-central1-b,us-central1-c \
    --num-nodes 1 \
    --machine-type e2-standard-4 \
    --disk-type pd-standard \
    --disk-size 50GB

gcloud container node-pools create gpu-pool \
    --cluster gpu-demo \
    --region us-central1 \
    --node-locations us-central1-a,us-central1-b,us-central1-c \
    --num-nodes 1 \
    --machine-type g2-standard-4 \
    --disk-type pd-ssd \
    --disk-size 100GB \
    --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
    --location-policy ANY \
    --spot
```
## ECK Pods
Three master pods and three data pods are provisioned.  The masters and one Kibana pod are mapped to the CPU-only nodes; the data pods are mapped to the GPU nodes via [taints/tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/).  Pods are kept separated on individual nodes and zones via [affinity](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity) rules.

The script below provisions the Elastic CRDs and pods. The loop awaits the Elasticsearch and Kibana objects to both be fully-provisioned.
```bash
kubectl create -f https://download.elastic.co/downloads/eck/3.2.0/crds.yaml > /dev/null 2>&1
kubectl apply -f https://download.elastic.co/downloads/eck/3.2.0/operator.yaml > /dev/null 2>&1
kubectl apply -f manifests
ES_STATUS=$(kubectl get elasticsearch -o=jsonpath='{.items[0].status.health}')
KB_STATUS=$(kubectl get kibana -o=jsonpath='{.items[0].status.health}')
while [[ $ES_STATUS != "green" ||  $KB_STATUS != "green" ]]
do  
  sleep 5
  ES_STATUS=$(kubectl get elasticsearch -o=jsonpath='{.items[0].status.health}')
  KB_STATUS=$(kubectl get kibana -o=jsonpath='{.items[0].status.health}')
done
```

# Data Set Generation
I use the [Faker](https://faker.readthedocs.io/en/master/) python lib to generate synthetic text fields.  Those fields are then sent to EIS for creation of embeddings.  Both items are written as a single document to a local [JSON Lines](https://jsonlines.org/) file.
```python
def create_data_file():
    fake = Faker()
    fake.seed_instance(12345)

    with open("data.jsonl", "w") as f:
        for _ in tqdm.tqdm(range(DATASET_SIZE // BATCH_SIZE)):
            paragraphs = fake.paragraphs(nb=BATCH_SIZE)
            embeddings = get_jina_embeddings(paragraphs)
            for paragraph, embedding in zip(paragraphs, embeddings):
                doc = {"paragraph": paragraph, "embedding": embedding}
                f.write(json.dumps(doc) + "\n")
            time.sleep(1) 
```

# Indexing
I subsequently read that jsonl file and bulk load it to the ECK cluster.  I create an index with a text and dense vector field corresponding to the Faker paragraph and Jina embedding.

```python
settings = {
    "index": {
        "number_of_shards": 3,
        "number_of_replicas": 1
    }
}
mappings = {
    "properties": {
        "paragraph": { "type": "text" },
        "embedding": {
            "type": "dense_vector",
            "dims": 1024,
            "index": True,
            "index_options": {
                "type": "int8_hnsw"
            }
        }
    }
}            

es.options(ignore_status=[404]).indices.delete(index=INDEX_NAME)
es.indices.create(index=INDEX_NAME, body={"settings": settings, "mappings": mappings})
```

# Testing
I wrote a simple Python script for performing an Elastic reindex command and then deriving the latency + throughput of that reindex operation.  This operation isolates the indexing process itself to demonstrate the acceleration derived from GPU use.  I forcibly clear cache, generated random index names and delete that index on each test round in order to eliminate the possibility of tests being influenced by caching.  I run one test with the GPUs disabled in Elastic and then another test with them enabled.
```python
def speed_test(tests=10):
    latencies = []
    throughputs = []

    for i in tqdm.tqdm(range(tests)):
        es.indices.clear_cache(index=INDEX_NAME)
        dest = f"{INDEX_NAME}_{uuid.uuid4()}"
        reindex_body = {
            "source": { "index": INDEX_NAME },
            "dest": { "index": dest }
        }
        response = es.reindex(slices="auto", body=reindex_body, wait_for_completion=False)
        task_id = response['task']
        latency, throughput = monitor_reindex(task_id)
        latencies.append(latency)
        throughputs.append(throughput)
        es.indices.delete(index=dest)
        if i < tests - 1:
            time.sleep(5)
    return latencies, throughputs
```


# Results
I create a graph comparing CPU vs GPU results using [pandas](https://pandas.pydata.org/) and [mathplotlib](https://matplotlib.org/).
![results](results.png)

# Source
https://github.com/joeywhelan/sm-gpu