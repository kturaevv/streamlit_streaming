up:
	docker container rm app || echo ""
	docker build -t app -f Dockerfile .
	docker run -d -p 8501:8501 --name app --restart=always app

down:
	docker container stop app

in:
	docker exec -it app sh