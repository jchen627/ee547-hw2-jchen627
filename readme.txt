MY FULL NAME:JIAXING CHEN EMAIL: jchen627@usc.edu
I have given the input format down blew

In Problem 2, we designed a bag-of-words autoencoder to learn paper embeddings. Each abstract is represented as a bag-of-words vector and passed through an encoder (V → hidden_dim → embedding_dim) to obtain a low-dimensional embedding. A decoder reconstructs the input (embedding_dim → hidden_dim → V), and the model is trained using binary cross-entropy loss. The resulting embeddings capture semantic information from the abstracts and are saved for downstream analysis.




Problem1 test：

./build.sh
./test.sh
./run.sh 9000
Get/papers command:
curl http://localhost:9000/papers
GET /papers/{2103.00112v3}:
curl "http://localhost:9000/papers/2103.00112v3"
GET /search?q={query}:
curl "http://localhost:9000/search?q=deep+learning"
GET /stats:
curl http://localhost:9000/stats



problem2 test:

./build.sh
./run.sh ../problem1/sample_data/papers.json output/
python train_embeddings.py ../problem1/sample_data/papers.json output/ --epochs 50 --batch_size 32




problem3 test:


east us 1
./test.sh
python aws_inspector.py --region us-east-1 --format json --output results/test.json
python aws_inspector.py --region us-east-1 --format table --output results/table_report.txt
