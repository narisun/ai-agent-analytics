"""Smoke tests: each route module exports an APIRouter and is importable."""
from fastapi import APIRouter


def test_chat_router_exists():
    from src.routes.chat import chat_router
    assert isinstance(chat_router, APIRouter)


def test_conversations_router_exists():
    from src.routes.conversations import conversations_router
    assert isinstance(conversations_router, APIRouter)


def test_health_router_exists():
    from src.routes.health import health_router
    assert isinstance(health_router, APIRouter)
