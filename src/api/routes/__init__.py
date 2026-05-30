"""API route handlers, one module per resource.

Each route module exports a single ``router`` object that
``src.api.main`` registers on the application. Keeping one resource per
file matches the standard FastAPI project structure and makes the
permissions / authentication story easier to audit later.
"""
