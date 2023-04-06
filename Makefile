up:
	docker container rm app || echo ""
	docker build -t app -f Dockerfile .
	docker run --rm -d -p 8501:8501 --name app  app

down:
	docker container stop app

in:
	docker exec -it app sh