
./bin/ollama serve &

pid=$!

sleep 5


echo "Pulling llama3 model"
ollama pull llama3.2:1b
ollama pull llama3.2:3b


wait $pid