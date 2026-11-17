# serve.py
from livereload import Server

server = Server()
# Watch the rendered output and refresh the browser when it changes
server.watch('_book/')
server.serve(root='_book/', port=5500)

# To preview:
# python docs/_tables/build_sectors_by_region.py
# quarto render --profile watermark
# uv run --with livereload python serve.py

# in browser:
# http://localhost:5500/


# To publish:
# quarto publish gh-pages --no-render