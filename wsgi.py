"""WSGI entry point for gunicorn"""
from viewer_app import app, start_app

start_app()


