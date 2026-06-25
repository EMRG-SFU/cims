# serve.py
from livereload import Server

server = Server()
# Watch the rendered output and refresh the browser when it changes
server.watch('_book/')
server.serve(root='_book/', port=5500)

# To use, run:
# uv run --with livereload python serve.py